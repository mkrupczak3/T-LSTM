# Time-Aware Long-Short Term Memory
(modifications of original code from https://github.com/illidanlab/T-LSTM)
Regularity of the duration between consecutive elements of a sequence is a property that does not always hold. An architecture that can overcome this irregularity is necessary to increase the prediction performance.

Time Aware LSTM (T-LSTM) was designed to handle irregular elapsed times. T-LSTM is proposed to incorporate the elapsed time
information into the standard LSTM architecture to be able to capture the temporal dynamics of sequential data with time irregularities. T-LSTM decomposes memory cell into short-term and long-term components, discounts the short-term memory content using a non-increasing function of the elapsed time, and then combines it with the long-term memory.

# Compatibility
Code is compatible with tensorflow version 1.6.0 and Pyhton 3.6.4.

# Modifications
1. Allow users to customize the number of encoders and decoders and the dimensions within each encoder/decoder.


## Forker's note for the layperson

Example of LSTM learning:
https://www.youtube.com/watch?v=Ipi40cb_RsI&t=254s

LSTM's have made a lot of news lately with things powering things like deepfakes and much of the AI work coming out of Google. They work really well with learning from things like video, audio, or games because there is a uniform time step or "tick" in between each piece of data.

T-LSTM's are a modified version of these neurons which are designed to allow them to better understand data where the time step is not uniform, such as in healthcare when doctor's visits can be either grouped close together or further apart.

T-LSTM's are currently only being used in healthcare and some other realms, but could prove to be very powerful model for analyzing many kinds of data.

In addition, using T-LSTM's with Generative Adversarial Networks (GANs) could allow for extremely powerful predictive capability on many different kinds of data. 

