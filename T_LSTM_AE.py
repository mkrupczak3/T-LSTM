# T-LSTM Autoencoder rewritten based on code written by Inci M. Baytas (https://github.com/illidanlab/T-LSTM)
# Duc Thanh Anh Luong
# University at Buffalo
# April, 2018
import tensorflow as tf
import math


class T_LSTM_AE(object):
    def __init__(self, n_encoders, n_decoders, input_dim_enc, input_dim_dec, hidden_dim_enc, hidden_dim_dec, output_dim_enc, output_dim_dec):
        self.n_encoders = n_encoders # number of encoders
        self.n_decoders = n_decoders # number of decoders

        assert self.n_encoders >= 1
        assert self.n_decoders >= 1
        
        self.hidden_dim_enc = hidden_dim_enc
        self.hidden_dim_dec = hidden_dim_dec

        assert len(self.hidden_dim_enc) == self.n_encoders
        assert len(self.hidden_dim_dec) == self.n_decoders
        assert hidden_dim_enc[-1] == hidden_dim_dec[0] # hidden dimension of last encoder should match the one in first decoder

        self.input_dim_enc = input_dim_enc
        self.input_dim_dec = input_dim_dec

        assert len(self.input_dim_enc) == self.n_encoders
        assert len(self.input_dim_dec) == self.n_decoders
        # dimension of input of first decoder is also the dimension of input of first encoder
        assert self.input_dim_dec[0] == self.input_dim_enc[0]
        
        self.output_dim_enc = output_dim_enc
        self.output_dim_dec = output_dim_dec
        
        assert len(self.output_dim_enc) == self.n_encoders-1
        assert len(self.output_dim_dec) == self.n_decoders

        # dimension of output of previous layer should match with dimension of input of current layer
        for i in range(1, self.n_encoders):
            assert self.output_dim_enc[i-1] == self.input_dim_enc[i]
        for i in range(1, self.n_decoders):
            assert self.output_dim_dec[i-1] == self.input_dim_dec[i]
        # dimension of output of last decoder should match with the dimension of input of first encoder
        assert self.output_dim_dec[-1] == self.input_dim_enc[0]

        self.init_parameters()
        
        # [batch size x seq length x input dim]
        self.input = tf.placeholder('float', shape=[None, None, self.input_dim_enc[0]])
        # [batch size x seq length]
        self.time = tf.placeholder('float', [None, None])

    def init_weights(self, input_dim, output_dim, name=None, std=1.0):
        return tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=std / math.sqrt(input_dim)), name=name)

    def init_bias(self, output_dim, name=None):
        return tf.Variable(tf.zeros([output_dim]), name=name)

    def init_parameters(self):
        self.init_parameters_encoder()
        self.init_parameters_decoder()
        self.init_parameters_output()

    def init_parameters_encoder(self):
        self.Wi_enc = [self.init_weights(self.input_dim_enc[i], self.hidden_dim_enc[i], name = 'Input_Hidden_weight_enc_' + str(i)) \
                       for i in range(self.n_encoders)]

        self.Ui_enc = [self.init_weights(self.hidden_dim_enc[i], self.hidden_dim_enc[i], name = 'Input_State_weight_enc_' + str(i)) \
                       for i in range(self.n_encoders)]
        
        self.bi_enc = [self.init_bias(self.hidden_dim_enc[i], name='Input_Hidden_bias_enc_' + str(i)) \
                       for i in range(self.n_encoders)]
        
        self.Wf_enc = [self.init_weights(self.input_dim_enc[i], self.hidden_dim_enc[i], name='Forget_Hidden_weight_enc_' + str(i)) \
                       for i in range(self.n_encoders)]
        
        self.Uf_enc = [self.init_weights(self.hidden_dim_enc[i], self.hidden_dim_enc[i], name = 'Forget_State_weight_enc_' + str(i)) \
                       for i in range(self.n_encoders)]
        
        self.bf_enc = [self.init_bias(self.hidden_dim_enc[i], name = 'Forget_Hidden_bias_enc_' + str(i)) for i in range(self.n_encoders)]

        self.Wog_enc = [self.init_weights(self.input_dim_enc[i], self.hidden_dim_enc[i], name = 'Output_Hidden_weight_enc_' + str(i)) \
                        for i in range(self.n_encoders)]
        
        self.Uog_enc = [self.init_weights(self.hidden_dim_enc[i], self.hidden_dim_enc[i], name='Output_State_weight_enc_' + str(i)) \
                        for i in range(self.n_encoders)]

        self.bog_enc = [self.init_bias(self.hidden_dim_enc[i], name='Output_Hidden_bias_enc_' + str(i)) \
                        for i in range(self.n_encoders)]
        
        self.Wc_enc = [self.init_weights(self.input_dim_enc[i], self.hidden_dim_enc[i], name = 'Cell_Hidden_weight_enc_' + str(i)) \
                       for i in range(self.n_encoders)]

        self.Uc_enc = [self.init_weights(self.hidden_dim_enc[i], self.hidden_dim_enc[i], name='Cell_State_weight_enc_' + str(i)) \
                       for i in range(self.n_encoders)]

        self.bc_enc = [self.init_bias(self.hidden_dim_enc[i], name='Cell_Hidden_bias_enc_' + str(i)) \
                       for i in range(self.n_encoders)]

        self.W_decomp_enc = [self.init_weights(self.hidden_dim_enc[i], self.hidden_dim_enc[i], name='Input_Hidden_weight_enc_' + str(i)) \
                             for i in range(self.n_encoders)]

        self.b_decomp_enc = [self.init_bias(self.hidden_dim_enc[i], name='Input_Hidden_bias_enc_' + str(i)) \
                             for i in range(self.n_encoders)]


    def init_parameters_decoder(self):
        self.Wi_dec = [self.init_weights(self.input_dim_dec[i], self.hidden_dim_dec[i], name = 'Input_Hidden_weight_dec_' + str(i)) \
                       for i in range(self.n_decoders)]

        self.Ui_dec = [self.init_weights(self.hidden_dim_dec[i], self.hidden_dim_dec[i], name = 'Input_State_weight_dec_' + str(i)) \
                       for i in range(self.n_decoders)]

        self.bi_dec = [self.init_bias(self.hidden_dim_dec[i], name='Input_Hidden_bias_dec_' + str(i)) \
                       for i in range(self.n_decoders)]

        self.Wf_dec = [self.init_weights(self.input_dim_dec[i], self.hidden_dim_dec[i], name='Forget_Hidden_weight_dec_' + str(i)) \
                       for i in range(self.n_decoders)]

        self.Uf_dec = [self.init_weights(self.hidden_dim_dec[i], self.hidden_dim_dec[i], name = 'Forget_State_weight_dec_' + str(i)) \
                       for i in range(self.n_decoders)]

        self.bf_dec = [self.init_bias(self.hidden_dim_dec[i], name = 'Forget_Hidden_bias_dec_' + str(i)) for i in range(self.n_decoders)]

        self.Wog_dec = [self.init_weights(self.input_dim_dec[i], self.hidden_dim_dec[i], name = 'Output_Hidden_weight_dec_' + str(i)) \
                        for i in range(self.n_decoders)]
        
        self.Uog_dec = [self.init_weights(self.hidden_dim_dec[i], self.hidden_dim_dec[i], name='Output_State_weight_dec_' + str(i)) \
                        for i in range(self.n_decoders)]

        self.bog_dec = [self.init_bias(self.hidden_dim_dec[i], name='Output_Hidden_bias_dec_' + str(i)) \
                        for i in range(self.n_decoders)]
        
        self.Wc_dec = [self.init_weights(self.input_dim_dec[i], self.hidden_dim_dec[i], name = 'Cell_Hidden_weight_dec_' + str(i)) \
                       for i in range(self.n_decoders)]

        self.Uc_dec = [self.init_weights(self.hidden_dim_dec[i], self.hidden_dim_dec[i], name='Cell_State_weight_dec_' + str(i)) \
                       for i in range(self.n_decoders)]

        self.bc_dec = [self.init_bias(self.hidden_dim_dec[i], name='Cell_Hidden_bias_dec_' + str(i)) \
                       for i in range(self.n_decoders)]

        self.W_decomp_dec = [self.init_weights(self.hidden_dim_dec[i], self.hidden_dim_dec[i], name='Input_Hidden_weight_dec_' + str(i)) \
                             for i in range(self.n_decoders)]
                             
        self.b_decomp_dec = [self.init_bias(self.hidden_dim_dec[i], name='Input_Hidden_bias_dec_' + str(i)) \
                             for i in range(self.n_decoders)]

    def init_parameters_output(self):
        # the last encoder doesn't need to produce output
        self.Wo_enc = [self.init_weights(self.hidden_dim_enc[i], self.output_dim_enc[i], name='Output_Layer_weight_enc_' + str(i)) \
                       for i in range(self.n_encoders - 1)]

        self.bo_enc = [self.init_bias(self.output_dim_enc[i], name='Output_Layer_bias_enc_' + str(i)) \
                       for i in range(self.n_encoders - 1)]

        self.Wo_dec = [self.init_weights(self.hidden_dim_dec[i], self.output_dim_dec[i], name='Output_Layer_weight_dec_' + str(i)) \
                       for i in range(self.n_decoders)]

        self.bo_dec = [self.init_bias(self.output_dim_dec[i], name='Output_Layer_bias_dec_' + str(i)) \
                       for i in range(self.n_decoders)]
        

    def T_LSTM_Encoder_Unit(self, enc_index):
        def f(prev_hidden_memory, concat_input):
            # shape of prev_hidden_state: batch_size x hidden_dim
            # shape of prev_cell:         batch_size x hidden_dim
            prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)

            batch_size = tf.shape(concat_input)[0]
            x = tf.slice(concat_input, [0,1], [batch_size, self.input_dim_enc[enc_index]])
            t = tf.slice(concat_input, [0,0], [batch_size, 1])

            # apply decaying function g(delta_t)
            T = self.map_elapse_time(t, self.hidden_dim_enc[enc_index])
            # shape of T: batch_size x hidden_dim

            # shape of C_ST: batch_size x hidden_dim
            C_ST = tf.nn.sigmoid(tf.matmul(prev_cell, self.W_decomp_enc[enc_index]) + self.b_decomp_enc[enc_index]) # sigmoid instead of tanh???
            # shape of C_ST_dis: batch_size x hidden_dim
            C_ST_dis = tf.multiply(T, C_ST)

            # if T is 0, then the weight is one
            # shape of prev_cell: batch_size x hidden_dim
            prev_cell = prev_cell - C_ST + C_ST_dis 


            # Input gate
            # shape of i: batch_size x hidden_dim
            i = tf.sigmoid(tf.matmul(x, self.Wi_enc[enc_index]) + tf.matmul(prev_hidden_state, self.Ui_enc[enc_index]) + self.bi_enc[enc_index])

            # Forget Gate
            # shape of f: batch_size x hidden_dim
            f = tf.sigmoid(tf.matmul(x, self.Wf_enc[enc_index]) + tf.matmul(prev_hidden_state, self.Uf_enc[enc_index]) + self.bf_enc[enc_index])

            # Output Gate
            # shape of o: batch_size x hidden_dim
            o = tf.sigmoid(tf.matmul(x, self.Wog_enc[enc_index]) + tf.matmul(prev_hidden_state, self.Uog_enc[enc_index]) + self.bog_enc[enc_index])

            # Candidate Memory Cell
            # shape of C: batch_size x hidden_dim
            C = tf.nn.tanh(tf.matmul(x, self.Wc_enc[enc_index]) + tf.matmul(prev_hidden_state, self.Uc_enc[enc_index]) + self.bc_enc[enc_index])

            # Current Memory cell
            # shape of Ct: batch_size x hidden_dim
            Ct = f * prev_cell + i * C

            # Current Hidden state
            # shape of current_hidden_state: batch_size x hidden_dim
            current_hidden_state = o * tf.nn.tanh(Ct)

            return tf.stack([current_hidden_state, Ct])
        return(f)

    def T_LSTM_Decoder_Unit(self, dec_index):
        def f(prev_hidden_memory, concat_input):
            prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)
            
            batch_size = tf.shape(concat_input)[0]
            
            x = tf.slice(concat_input, [0, 1], [batch_size, self.input_dim_dec[dec_index]])
            t = tf.slice(concat_input, [0, 0], [batch_size, 1])
            
            # Map elapse time in days or months
            T = self.map_elapse_time(t, self.hidden_dim_dec[dec_index])
            C_ST = tf.nn.sigmoid(tf.matmul(prev_cell, self.W_decomp_dec[dec_index]) + self.b_decomp_dec[dec_index])
            C_ST_dis = tf.multiply(T, C_ST)
            
            # if T is 0, then the weight is one
            prev_cell = prev_cell - C_ST + C_ST_dis
            
            # Input gate
            i = tf.sigmoid(tf.matmul(x, self.Wi_dec[dec_index]) + tf.matmul(prev_hidden_state, self.Ui_dec[dec_index]) + self.bi_dec[dec_index])

            # Forget Gate
            f = tf.sigmoid(tf.matmul(x, self.Wf_dec[dec_index]) + tf.matmul(prev_hidden_state, self.Uf_dec[dec_index]) + self.bf_dec[dec_index])

            # Output Gate
            o = tf.sigmoid(tf.matmul(x, self.Wog_dec[dec_index]) + tf.matmul(prev_hidden_state, self.Uog_dec[dec_index]) + self.bog_dec[dec_index])

            # Candidate Memory Cell
            C = tf.nn.tanh(tf.matmul(x, self.Wc_dec[dec_index]) + tf.matmul(prev_hidden_state, self.Uc_dec[dec_index]) + self.bc_dec[dec_index])

            # Current Memory cell
            Ct = f * prev_cell + i * C
            
            # Current Hidden state
            current_hidden_state = o * tf.nn.tanh(Ct)

            return tf.stack([current_hidden_state, Ct])
        return(f)

    def get_encoder_states(self): # Returns all hidden states for the samples in a batch
        batch_size = tf.shape(self.input)[0]
        scan_input_ = tf.transpose(self.input, perm=[2, 0, 1]) # input_dim x batch_size x seq_length
        scan_input = tf.transpose(scan_input_) #scan input is [seq_length x batch_size x input_dim]
        scan_time = tf.transpose(self.time) # scan_time [seq_length x batch_size]
        
        # make scan_time [seq_length x batch_size x 1]
        scan_time = tf.reshape(scan_time, [tf.shape(scan_time)[0],tf.shape(scan_time)[1],1])
        
        for i in range(self.n_encoders-1):
            concat_input = tf.concat([scan_time, scan_input],2) # [seq_length x batch_size x input_dim+1]
            initial_hidden = tf.zeros([batch_size, self.hidden_dim_enc[i]], tf.float32) #np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
            ini_state_cell = tf.stack([initial_hidden, initial_hidden]) # initial hidden and initial memory
            # run T-LSTM Encoder on concat_input, starting from ini_state_cell (initial hidden units and initial memory)
            f_encoder = self.T_LSTM_Encoder_Unit(i)
            packed_hidden_states = tf.scan(f_encoder, concat_input, initializer=ini_state_cell, name='encoder_states_' + str(i))
            encoder_states = packed_hidden_states[:, 0, :, :]
            encoder_cells = packed_hidden_states[:, 1, :, :]

            # get the output from ith encoder 
            f_output = self.get_output(encoder = True, index = i)
            scan_input = tf.map_fn(f_output, encoder_states)

        # run the last encoder
        concat_input = tf.concat([scan_time, scan_input],2) # [seq_length x batch_size x input_dim+1]
        initial_hidden = tf.zeros([batch_size, self.hidden_dim_enc[-1]], tf.float32) #np.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        ini_state_cell = tf.stack([initial_hidden, initial_hidden]) # initial hidden and initial memory
        # run T-LSTM Encoder on concat_input, starting from ini_state_cell (initial hidden units and initial memory)
        f_encoder = self.T_LSTM_Encoder_Unit(self.n_encoders - 1)
        packed_hidden_states = tf.scan(f_encoder, concat_input, initializer=ini_state_cell, name='encoder_states_' + str(self.n_encoders - 1))
        encoder_states = packed_hidden_states[:, 0, :, :]
        encoder_cells = packed_hidden_states[:, 1, :, :]
        return(encoder_states, encoder_cells)


    def get_representation(self):
        last_encoder_states, last_encoder_cells = self.get_encoder_states()
        # We need the last hidden state of the encoder
        representation = last_encoder_states[-1, :, :]
        decoder_ini_cell = last_encoder_cells[-1, :, :]
        return representation, decoder_ini_cell

    def get_output(self, encoder, index):
        def internal_get_output(state):
            if encoder == True:
                output = tf.matmul(state, self.Wo_enc[index]) + self.bo_enc[index]    
            else:
                output = tf.matmul(state, self.Wo_dec[index]) + self.bo_dec[index]
            return(output)
        return(internal_get_output)
        

    def get_decoder_states(self):
        batch_size = tf.shape(self.input)[0]
        seq_length = tf.shape(self.input)[1]
        scan_input_ = tf.transpose(self.input, perm=[2, 0, 1])
        scan_input_ = tf.transpose(scan_input_)  # scan input is [seq_length x batch_size x input_dim]
        z = tf.zeros([1, batch_size, self.input_dim_dec[0]], dtype=tf.float32)
        scan_input = tf.concat([scan_input_,z],0)
        scan_input = tf.slice(scan_input, [1,0,0],[seq_length ,batch_size, self.input_dim_dec[0]])
        scan_input = tf.reverse(scan_input, [0])
        scan_time_ = tf.transpose(self.time)  # scan_time [seq_length x batch_size]
        z2 = tf.zeros([1, batch_size], dtype=tf.float32)
        scan_time = tf.concat([scan_time_, z2],0)
        scan_time = tf.slice(scan_time, [1,0],[seq_length ,batch_size])
        scan_time = tf.reverse(scan_time, [0])
        
        # make scan_time [seq_length x batch_size x 1]
        scan_time = tf.reshape(scan_time, [tf.shape(scan_time)[0], tf.shape(scan_time)[1], 1])
        
        for i in range(self.n_decoders):
            concat_input = tf.concat([scan_time, scan_input],2)  # [seq_length x batch_size x input_dim+1]
            # prepare first hidden and cell
            if i == 0:
                initial_hidden, initial_cell = self.get_representation()
                ini_state_cell = tf.stack([initial_hidden, initial_cell])        
            else:
                initial_hidden = tf.zeros([batch_size, self.hidden_dim_dec[i]], tf.float32)
                ini_state_cell = tf.stack([initial_hidden, initial_hidden])
    
            # run T-LSTM Decoder on concat_input, starting from ini_state_cell (initial hidden units and initial memory)
            f_decoder = self.T_LSTM_Decoder_Unit(i)
            packed_hidden_states = tf.scan(f_decoder, concat_input, initializer=ini_state_cell, name='decoder_states_' + str(i))
            decoder_states = packed_hidden_states[:, 0, :, :]

            # get the output from ith decoder 
            f_output = self.get_output(encoder = False, index = i)
            scan_input = tf.map_fn(f_output, decoder_states)
            
        return(decoder_states)

    def get_decoder_outputs(self): # Returns the output of only the last time step
        decoder_states = self.get_decoder_states()
        # get output from last decoder
        f_output = self.get_output(encoder = False, index = self.n_decoders - 1)
        all_outputs = tf.map_fn(f_output, decoder_states)
        
        reversed_outputs = tf.reverse(all_outputs, [0])
        outputs_ = tf.transpose(reversed_outputs, perm=[2, 0, 1])
        outputs = tf.transpose(outputs_)
        return outputs
    
    # get reconstuction loss
    def get_reconstruction_loss(self):
        outputs = self.get_decoder_outputs() # get the output from decoder
        loss = tf.reduce_mean(tf.square(self.input - outputs)) # square of difference between input and output
        return loss

    # heuristic decaying function g(delta_t)
    def map_elapse_time(self, t, dim):

        c1 = tf.constant(1, dtype=tf.float32)
        c2 = tf.constant(2.7183, dtype=tf.float32)

        T = tf.div(c1, tf.log(t + c2), name='Log_elapse_time')
        # T = tf.div(c1, tf.add(t , c1), name='Log_elapse_time')

        Ones = tf.ones([1, dim], dtype=tf.float32)

        T = tf.matmul(T, Ones)

        return T


