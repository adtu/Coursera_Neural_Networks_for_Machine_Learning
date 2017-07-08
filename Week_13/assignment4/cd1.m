function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    % error('not yet implemented');
%{
    % original CD1 implementation
    visible_state = visible_data;
    hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_state);
    hidden_state = sample_bernoulli(hidden_probability);
    d_G_by_rbm_w_0 = configuration_goodness_gradient(visible_state, hidden_state);

    % visible state reconstruction
    visible_probability = hidden_state_to_visible_probabilities(rbm_w, hidden_state);
    visible_state = sample_bernoulli(visible_probability);

    % generated hidden state from visible state reconstructed
    hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_state);
    hidden_state = sample_bernoulli(hidden_probability);
    d_G_by_rbm_w_1 = configuration_goodness_gradient(visible_state, hidden_state);

    ret = d_G_by_rbm_w_0 - d_G_by_rbm_w_1;
%}

%{
    % improved CD1 implementation
    visible_state = visible_data;
    hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_state);
    hidden_state = sample_bernoulli(hidden_probability);
    d_G_by_rbm_w_0 = configuration_goodness_gradient(visible_state, hidden_state);

    % visible state reconstruction
    visible_probability = hidden_state_to_visible_probabilities(rbm_w, hidden_state);
    visible_state = sample_bernoulli(visible_probability);

    % generated hidden state from visible state reconstructed
    hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_state);
    hidden_state = hidden_probability;
    d_G_by_rbm_w_1 = configuration_goodness_gradient(visible_state, hidden_state);

    ret = d_G_by_rbm_w_0 - d_G_by_rbm_w_1;
%}


    % improved CD1 implementation with real-value transformed to binary value
    % real-value data transformation to binary value data by sampling
    visible_data = sample_bernoulli(visible_data);

    % sample hidden state
    visible_state = visible_data;
    hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_state);
    hidden_state = sample_bernoulli(hidden_probability);
    d_G_by_rbm_w_0 = configuration_goodness_gradient(visible_state, hidden_state);

    % visible state reconstruction
    visible_probability = hidden_state_to_visible_probabilities(rbm_w, hidden_state);
    visible_state = sample_bernoulli(visible_probability);

    % generated hidden state from visible state reconstructed
    hidden_probability = visible_state_to_hidden_probabilities(rbm_w, visible_state);
    hidden_state = hidden_probability;
    d_G_by_rbm_w_1 = configuration_goodness_gradient(visible_state, hidden_state);

    ret = d_G_by_rbm_w_0 - d_G_by_rbm_w_1;



end
