function [A, b, x0, x_true, b_tilde,idx] = generate_system(options)
    % GENERATE_SYSTEM generates a synthetic linear system.
    % ** translated from Python to MATLAB by Gemini
    %
    % Options (passed as an argument block):
    %   options.m                   - Number of rows (default: 500)
    %   options.n                   - Number of columns (default: 10)
    %   options.low                 - Lower bound for uniform dist (default: -10)
    %   options.high                - Upper bound for uniform dist (default: 10)
    %   options.corrupt             - Boolean, corrupt b? (default: true)
    %   options.corrupt_beta        - Fraction of indices to corrupt (default: 0.01)
    %   options.corrupt_magnitude   - Mean of the corruption dist (default: 100)
    %   options.rng_seed            - Seed for random number generator (default: 42)
    % 
    % How to use: [A, b, x0, x_true, b_tilde] = generate_system('m', 1000, 'corrupt', true);

    arguments
        options.m {mustBeNumeric} = 1000
        options.n {mustBeNumeric} = 50
        options.low {mustBeNumeric} = -10
        options.high {mustBeNumeric} = 10
        options.corrupt {mustBeNumericOrLogical} = true
        options.corrupt_beta {mustBeNumeric} = 0.20
        options.corrupt_magnitude {mustBeNumeric} = 100;
        options.rng_seed {mustBeNumeric} = 42
        options.type {mustBeText} = 'uniform'
        options.scale {mustBeNumeric} = 1
    end

    % Extract arguments for easier reading
    m = options.m;
    n = options.n;
    low = options.low;
    high = options.high;
    corrupt = options.corrupt;
    corrupt_beta = options.corrupt_beta;
    corrupt_magnitude = options.corrupt_magnitude;
    type = options.type;

    % Fix the random generator
    rng(options.rng_seed, 'twister');

    if strcmp(type, 'normal')
        % Generate A and x_true from a scaled normal distribution
        A = options.scale * randn(m, n);
        x_true = options.scale * randn(n, 1);
        disp('normal')
    elseif strcmp(type, 'uniform')
        % Generate A and x_true from a uniform distribution [low, high]
        % MATLAB rand generates [0, 1]. Scale to [low, high].
        range = high - low;
        A = range * rand(m, n) + low;
        x_true = range * rand(n, 1) + low;
        disp('uniform')
    end
    
    % True uncorrupted b
    b = A * x_true;

    if corrupt
        k = round(corrupt_beta * m);
        c = zeros(m, 1);
        
        if k > 0
            % Choose k unique indices without replacement
            idx = randperm(m, k);
            
            % Generate normal distribution: mean = magnitude, std_dev = 1.0
            corrupt_values = corrupt_magnitude + 1.0 * randn(k, 1);
            
            c(idx) = corrupt_values;
        end
    else
        c = zeros(m, 1);
    end
    
    b_tilde = b + c;

    % Initialize the solution x (as a column vector)
    x0 = zeros(n, 1);

end