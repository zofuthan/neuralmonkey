#tests: lint

import tensorflow as tf
import numpy as np

from neuralmonkey.nn.ortho_gru_cell import OrthoGRUCell
from neuralmonkey.vocabulary import START_TOKEN
from neuralmonkey.logging import log
from neuralmonkey.decoders.output_projection import no_deep_output
from neuralmonkey.nn.projection import linear


# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-statements
# Big decoder cannot be simpler. Not sure if refactoring
# it into smaller units would be helpful
# Some locals may be turned to attributes

# pylint: disable=no-member
# We are using __dict__.update
class Decoder(object):
    """A class that manages parts of the computation graph that are
    used for the decoding.
    """

    kwargs_defaults = {
        "output_projection": no_deep_output,
        "max_output_len": 20,
        "embedding_size": 200,
        "dropout_keep_prob": 1.0,
        "project_encoder_outputs": False,
        "use_attention": False,
        "reuse_word_embeddings": False, #TODO change this to None or encoder
        "rnn_size": 200}

    def __init__(self, encoders, vocabulary, data_id, name, **kwargs):
        """Creates a new instance of the decoder

        Arguments:
            encoders: List of encoders whose outputs will be decoded
            vocabulary: Output vocabulary
            data_id: Identifier of the data series fed to this decoder

        Keyword arguments:
            embedding_size: Size of embedding vectors. Default 200
            max_output_len: Maximum length of the output. Default 20
            rnn_size: When projection is used or when no encoder is supplied,
                this is the size of the projected vector.
            dropout_keep_prob: Dropout keep probability. Default 1 (no dropout)
            use_attention: Boolean flag that indicates whether to use attention
                from encoders
            reuse_word_embeddings: Boolean flag specifying whether to
                reuse word embeddings. If True, word embeddings
                from the first encoder will be used
            project_encoder_outputs: Boolean flag whether to project output
                states of encoders
        """
        self.encoders = encoders
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.name = name

        # set all default arguments:
        self.__dict__.update(Decoder.kwargs_defaults)
        # replace default values with values from kwargs:
        self.__dict__.update(kwargs)

        # check for unknown kwarg arguments:
        for option, val in kwargs.items():
            if option not in Decoder.kwargs_defaults:
                log("Warning: Unknown kwarg setting {}, value {}"
                    .format(option, val), color="red")

        if self.reuse_word_embeddings:
            self.embedding_size = self.encoders[0].embedding_size

            if "embedding_size" in kwargs:
                log("Warning: Overriding embedding_size parameter with reused"
                    " embeddings from the encoder.", color="red")

        if len(self.encoders) > 0 and not self.project_encoder_outputs:
            if "rnn_size" in kwargs:
                log("Warning: rnn_size attribute will not be used "
                    "without encoder projection!", color="red")

            self.rnn_size = sum(e.encoded.get_shape()[1].value
                                for e in self.encoders)


        log("Initializing decoder, name: '{}'".format(self.name))


        with tf.variable_scope(name) as scope:

            ### Learning step
            ### TODO was here only because of scheduled sampling.
            ### needs to be refactored out
            self.learning_step = tf.get_variable(
                "learning_step", [], initializer=tf.constant_initializer(0),
                trainable=False)

            self._create_placeholder_nodes()
            self._create_initial_state()
            self._create_embedding_matrix()

            embedded_train_inputs = self._embed(self.train_inputs[:-1])
            embedded_go_symbols = self._embed(self.go_symbols)

            self.train_rnn_outputs, _, \
                self.train_logits = self._decoding_function(
                    embedded_train_inputs, runtime_mode=False)

            assert not scope.reuse
            # Use the same variables for runtime decoding!
            scope.reuse_variables()
            assert scope.reuse

            # runtime methods and objects are used when no ground truth is
            # provided (such as during testing)
            self.runtime_rnn_outputs, self.runtime_rnn_states, \
                self.runtime_logits = self._decoding_function(
                    embedded_go_symbols, runtime_mode=True)

            # NOTE From this point onwards, the variables in this scope
            # REMAIN reused, so no further creation of variables is allowed

            # TODO instead of lists, work with time x batch tensors here

            self.train_logprobs = [tf.nn.log_softmax(l)
                                   for l in self.train_logits]

            self.runtime_logprobs = [tf.nn.log_softmax(l)
                                     for l in self.runtime_logits]

            self.decoded = [tf.argmax(l[:, 1:], 1) + 1
                            for l in self.runtime_logits]

            self.train_loss = tf.nn.seq2seq.sequence_loss(
                self.train_logits, self.train_targets,
                tf.unpack(self.train_padding), self.vocabulary_size) * 100

            self.runtime_loss = tf.nn.seq2seq.sequence_loss(
                self.runtime_logits, self.train_targets,
                tf.unpack(self.train_padding), self.vocabulary_size) * 100

            self.cross_entropies = tf.nn.seq2seq.sequence_loss_by_example(
                self.train_logits, self.train_targets,
                tf.unpack(self.train_padding), self.vocabulary_size)

            self._init_summaries()

        log("Decoder initialized.")

    @property
    def vocabulary_size(self):
        return len(self.vocabulary)

    @property
    def cost(self):
        return self.train_loss


    def top_k_runtime_logprobs(self, k_best):
        """Return the top runtime log probabilities calculated from runtime
        logits.

        Arguments:
            k_best: How many output items to return
        """
        ## the array is of tuples ([values], [indices])
        return [tf.nn.top_k(p, k_best) for p in self.runtime_logprobs]


    def _create_placeholder_nodes(self):
        """Creates placeholder nodes in the computation graph"""

        self.train_mode = tf.placeholder(
            tf.bool, shape=[], name="mode_placeholder")

        self.train_inputs = tf.placeholder(
            tf.int64, [self.max_output_len + 2, None],
            name="decoder_input_placeholder")

        batch_size = tf.shape(self.train_inputs)[1]
        self.train_targets = tf.unpack(self.train_inputs[1:])

        self.train_padding = tf.placeholder(
            tf.float32, [self.max_output_len + 1, None],
            name="decoder_padding_placeholder")

        # Explanation of the lines below:
        #   - inner expand_dims converts scalar batch size to
        #     a 1-D Tensor which is needed for tf.fill
        #   - tf.fill copies go_symbol batch-size-times.
        #   - outer expand_dims convert 1-D tensor of go symbols in batch
        #     to (1 x batch)-shaped tensor.  The same could be achieved with
        #     tf.reshape

        go_symbol_idx = self.vocabulary.get_word_index(START_TOKEN)
        self.go_symbols = tf.expand_dims(
            tf.fill(tf.expand_dims(batch_size, 0), go_symbol_idx), 0)


    def _dropout(self, variable):
        """Performs dropout on the variable

        Arguments:
            variable: The variable to be dropped out.
        """
        # Maintain clean graph - no dropout op when there is none applied
        if self.dropout_keep_prob == 1.0:
            return variable

        train_mode_selector = tf.fill(tf.shape(variable)[:1], self.train_mode)
        dropped_value = tf.nn.dropout(variable, self.dropout_keep_prob)
        return tf.select(train_mode_selector, dropped_value, variable)


    def _encoder_projection(self, encoded_states):
        """Creates a projection of concatenated encoder states
        and applies a tanh activation

        Arguments:
            encoded_states: Tensor of concatenated states of input encoders
                            (batch x sum(states))
        """
        input_size = encoded_states.get_shape()[1].value
        weights = tf.get_variable(
            "encoder_projection_W", [input_size, self.rnn_size],
            initializer=tf.random_normal_initializer(stddev=0.01))

        biases = tf.get_variable(
            "encoder_projection_b",
            initializer=tf.zeros_initializer([self.rnn_size]))

        dropped_input = self._dropout(encoded_states)
        return tf.tanh(tf.matmul(dropped_input, weights) + biases)


    def _create_initial_state(self):
        """Construct the part of the computation graph that computes the initial
        state of the decoder."""

        if len(self.encoders) == 0:
            return tf.zeros([self.rnn_size])

        encoders_out = tf.concat(1, [e.encoded for e in self.encoders])

        if self.project_encoder_outputs:
            encoders_out = self._encoder_projection(encoders_out)

        self.initial_state = self._dropout(encoders_out)

        ## Broadcast the initial state to the whole batch if needed
        ## CHANGE: REFACTORED + EXPAND DIMS USED INSTEAD OF [:1]. ALSO, THE
        ## DIMENSIONS ARE DIFFERENT
        if len(self.initial_state.get_shape()) == 1:
            assert self.initial_state.get_shape()[0].value == self.rnn_size

            batch_size = tf.shape(self.train_inputs[0])[1]
            tiles = tf.tile(self.initial_state, tf.expand_dims(batch_size, 0))

            self.initial_state = tf.reshape(tiles, [-1, self.rnn_size])



    def _create_embedding_matrix(self):
        """Create variables and operations for embedding of input words

        If we are reusing word embeddings, this function takes the embedding
        matrix from the first encoder
        """
        # NOTE In the Bahdanau paper, they say they initialized some weights
        # as orthogonal matrices, some by sampling from gauss distro with
        # stddev=0.001 and all other weight matrices as gaussian with
        # stddev=0.01. Embeddings were not among any of the special cases so
        # I assume that they initialized them as any other weight matrix.

        if self.reuse_word_embeddings:
            self.embedding_matrix = self.encoders[0].word_embeddings
        else:
            self.embedding_matrix = tf.get_variable(
                "word_embeddings", [self.vocabulary_size, self.embedding_size],
                initializer=tf.random_normal_initializer(stddev=0.01))


    def _embed(self, inputs):
        """Embed the input using the embedding matrix and apply dropout

        Arguments:
            inputs: The Tensor to be embedded and dropped out.
        """
        embedded = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
        return self._dropout(embedded)




    def _get_rnn_cell(self):
        """Returns a RNNCell object for this decoder"""
        return OrthoGRUCell(self.rnn_size)


    def _collect_attention_objects(self):
        """Collect attention objects from encoders."""
        if not self.use_attention:
            return []
        return [e.attention_object for e in self.encoders if e.attention_object]




    def _loop_function(self, rnn_output):
        """Basic loop function. Projects state to logits, take the
        argmax of the logits, embed the word and perform dropout on the
        embedding vector.

        Arguments:
            rnn_output: The output of the decoder RNN
        """
        output_activation = self._logit_function(rnn_output)
        previous_word = tf.argmax(output_activation, 1)
        input_embedding = tf.nn.embedding_lookup(self.embedding_matrix,
                                                 previous_word)
        return self._dropout(input_embedding)


    def _logit_function(self, rnn_output):
        """Compute logits on the vocabulary given the state

        This variant simply linearly project the vectors to fit
        the size of the vocabulary

        Arguments:
            rnn_output: the output of the decoder RNN
                        (after output projection)

        Returns:
            A Tensor of shape batch_size x vocabulary_size
        """
        return linear(self._dropout(rnn_output), self.vocabulary_size)


    def _decoding_function(self, inputs, runtime_mode):
        """Run the decoder RNN.

        Arguments:
            inputs: The decoder inputs. If runtime_mode=True, only the first
                    input is used.
            runtime_mode: Boolean flag whether the decoder is running in
                          runtime mode (with loop function).
        """
        cell = self._get_rnn_cell()
        att_objects = self._collect_attention_objects()
        initial_state = self.initial_state

        with tf.variable_scope("decoding_function"):

            ## CHANGE:
            ## EMBEDDED INPUTS NOW OF SHAPE TIME x BATCH x RNN_SIZE
            ## EMBEDDED GO_SYMBOLS ARE 1 x BATCH x RNN_SIZE

            contexts = [a.attention(initial_state)
                        for a in att_objects]

            output = self.output_projection(
                inputs[0], initial_state, contexts)

            _, state = cell(
                tf.concat(1, [inputs[0]] + contexts), initial_state)

            logit = self._logit_function(output)

            output_logits = [logit]
            rnn_outputs = [output]
            rnn_states = [initial_state, state]

            tf.get_variable_scope().reuse_variables()

            for step in range(1, self.max_output_len + 1):

                if runtime_mode:
                    # NOTE loop function must never leave this scope
                    # because the _logit_function is scope-sensitive,
                    # meaning it would create a new set of parameters
                    # in a different scope
                    current_input = self._loop_function(output)
                else:
                    current_input = inputs[step]

                ## N-th decoding step
                contexts = [a.attention(state) for a in att_objects]
                output = self.output_projection(
                    current_input, state, contexts)
                _, state = cell(
                    tf.concat(1, [current_input] + contexts), state)

                logit = self._logit_function(output)

                logit = self._logit_function(output)

                output_logits.append(logit)
                rnn_outputs.append(output)
                rnn_states.append(state)

            if runtime_mode:
                for i, a in enumerate(att_objects):
                    time = self.max_output_len + 1
                    attentions = a.attentions_in_time[-time:]
                    alignments = tf.expand_dims(tf.transpose(
                        tf.pack(attentions), perm=[1, 2, 0]), -1)

                    tf.image_summary(
                        "attention_{}".format(i), alignments,
                        collections=["summary_val_plots"],
                        max_images=256)

        return rnn_outputs, rnn_states, output_logits


    def _init_summaries(self):
        """Initialize the summaries of the decoder

        TensorBoard summaries are collected into the following
        collections:

        - summary_train: collects statistics from the train-time
        """
        tf.scalar_summary("train_loss_with_decoded_inputs",
                          self.runtime_loss,
                          collections=["summary_train"])

        tf.scalar_summary("train_optimization_cost", self.train_loss,
                          collections=["summary_train"])


    def feed_dict(self, dataset, train=False):
        """Populate the feed dictionary for the decoder object

        Decoder placeholders:

            ``decoder_{x} for x in range(max_output+2)``
            Training data placeholders. Starts with <s> and ends with </s>

            ``decoder_padding_weights{x} for x in range(max_output+1)``
            Weights used for padding. (Float) tensor of ones and zeros.
            This tensor is one-item shorter than the other one since the
            decoder does not produce the first <s>.

            ``dropout_placeholder``
            Scalar placeholder for dropout probability.
            Has value 'dropout_keep_prob' from the constructor or 1
            in case we are decoding at run-time
        """
        # pylint: disable=invalid-name
        # fd is the common name for feed dictionary
        fd = {}
        fd[self.train_mode] = train
        sentences = dataset.get_series(self.data_id, allow_none=True)

        if sentences is not None:
            inputs, weights = self.vocabulary.sentences_to_tensor(
                sentences, self.max_output_len)

            fd[self.train_padding] = weights
            fd[self.train_inputs] = inputs
        else:
            start_token_index = self.vocabulary.get_word_index(
                START_TOKEN)

            fd[self.train_inputs[0]] = np.repeat(start_token_index,
                                                 len(dataset))
            for placeholder in self.train_padding:
                fd[placeholder] = np.ones(len(dataset))

        return fd
