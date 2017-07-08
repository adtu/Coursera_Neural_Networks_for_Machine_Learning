function ret = partition(rbm_w)
    % Calculate the partition function using computationally friendly method
    num_hidden = size(rbm_w, 1);
    num_visible = size(rbm_w, 2);
    % here suppose num_visible is a small number
    Z = 0;
    for i = 0:(2^num_hidden-1)
	% disp(i);
        % generate non-duplicated visible state vector
        hidden_state = de2bi(i, num_hidden);
	product = 1;
	for j = 1:num_visible
	    visible_state = zeros(1, num_visible);
	    visible_state(j) = 1;
	    configuration_goodness = transpose(hidden_state) * visible_state .* rbm_w;
	    product = product * (exp(0) + exp(sum(configuration_goodness(:))));
	end
	Z = Z + product;
    end
    ret = log(Z);
end

