% script_betasearch.m
% For every weight function, run the algorithm across a range of beta values
% and plot two subplots: coarse beta search (group1) and fine beta search (group2).

% clear; clc;

%% ── Parameters ───────────────────────────────────────────────────────────────
m           = 1000;
n           = 50;
corrupt_mag = 0;
sys_seed    = 32;
num_seeds   = 3;
t_max       = 10000;
tol         = 1e-12;
eta         = 1.0;
output_dir  = 'figs_1009';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

%% ── Beta groups ──────────────────────────────────────────────────────────────
group1    = round(0.1:0.2:0.9, 4);   % coarse: 0.1, 0.2, ..., 0.7
group2    = round(0.5:0.02:0.6, 4);  % fine:   0.50, 0.51, ..., 0.60
testgroup = union(group1, group2);
num_betas = numel(testgroup);

% Indices of group1 and group2 within testgroup (for plotting)
[~, g1_idx] = ismember(group1, testgroup);
[~, g2_idx] = ismember(group2, testgroup);

%% ── Weight functions ─────────────────────────────────────────────────────────
q = 0.8;
% 
% W_funcs  = {@(r,n) qrkW(r, q, []), @fChoice6W, @fChoice1W, @fChoice4W, @fChoice5W, ...
%             @fChoice2W, @fChoice3W};
% W_labels = {sprintf('$\\mathrm{QRK}\\,(q=%.2f)$', q), ...
%             '$f_6(r)=e^{-r^2/\mathrm{med}(|r|)}$', ...
%             '$f_1(r)=e^{-r^2/\mathrm{med}(|r|)^2}$', ...
%             '$f_4(r)=e^{-r^2/\sigma_{\mathrm{MAD}}}$', ...
%             '$f_5(r)=e^{-r^2/\sigma_{\mathrm{MAD}}^2}$', ...
%             '$1/r^2$', '$1/|r|$'};
% W_names  = {sprintf('QRK_q%.2f', q), 'f6', 'f1', 'f4', 'f5', 'f2', 'f3'};

% W_funcs  = {@fChoice6W, @fChoice1W, @fChoice4W, @fChoice5W};
% W_labels = {'$f_6(r)=e^{-r^2/\mathrm{med}(|r|)}$', ...
%             '$f_1(r)=e^{-r^2/\mathrm{med}(|r|)^2}$', ...
%             '$f_4(r)=e^{-r^2/\sigma_{\mathrm{MAD}}}$', ...
%             '$f_5(r)=e^{-r^2/\sigma_{\mathrm{MAD}}^2}$'};
% W_names  = {'f6', 'f1', 'f4', 'f5'};

W_funcs  = {@fChoice1W};
W_labels = {'$f_1(r)=e^{-r^2/\mathrm{med}(|r|)^2}$'};
W_names  = {'f1'};
num_methods = numel(W_funcs);

%% ── Storage: num_methods × num_betas ────────────────────────────────────────
xh_med  = cell(num_methods, num_betas);
xh_min  = cell(num_methods, num_betas);
xh_max  = cell(num_methods, num_betas);

%% ── Run experiments ──────────────────────────────────────────────────────────
for mi = 1:num_methods
    for bi = 1:num_betas
        beta = testgroup(bi);

        [A, ~, ~, x_true, b_tilde, ~] = generate_system( ...
            'corrupt_beta', beta, 'rng_seed', sys_seed, ...
            'n', n, 'm', m, 'corrupt_magnitude', corrupt_mag, ...
            'type', 'normal');
        x0 = zeros(size(x_true));

        fprintf('Running %s, beta=%.2f ...\n', W_names{mi}, beta);
        cand_iters = zeros(1, num_seeds);
        cand = cell(num_seeds, 1);

        for si = 1:num_seeds
            rng(41 + si, 'twister');
            p_func = @l2erk1p;
            if mi == 1, p_func = @qrkp; end   % QRK is always method 1
            [~, resnorm, xh, iter, ~, rf, sel] = wrk(A, b_tilde, x0, x_true, ...
                eta, t_max, tol, W_funcs{mi}, p_func, n, false);
            cand_iters(si) = iter;
            cand{si} = struct('xh', xh, 'resnorm', resnorm, 'iter', iter);
        end

        [~, med_i] = min(abs(cand_iters - median(cand_iters)));
        [~, min_i] = min(cand_iters);
        [~, max_i] = max(cand_iters);

        xh_med{mi, bi} = cand{med_i}.xh;
        xh_min{mi, bi} = cand{min_i}.xh;
        xh_max{mi, bi} = cand{max_i}.xh;

        fprintf('  min=%d  median=%d  max=%d\n', ...
            min(cand_iters), round(median(cand_iters)), max(cand_iters));
    end
end

%% ── Plot settings ────────────────────────────────────────────────────────────
% W_names order: f6, f1, f4, f5
% line_styles = {':', '--', '-', '-.'};
line_styles = {'-'};

%% ── Plot: one figure per method, two subplots (coarse / fine beta) ───────────
for mi = 1:num_methods
    lsty = line_styles{mi};

    % Separate full-range colormaps per panel — each panel spans full hue range
    h1 = rgb2hsv(hsv(numel(g1_idx))); h1(:,3) = h1(:,3) * 0.75; c1 = hsv2rgb(h1);

    % Shift g2 hue cycle so β=0.5 (k=1 in g2) matches β=0.5's color in g1
    g1_beta05_k  = find(ismember(testgroup(g1_idx), 0.5), 1);
    hue_offset   = (g1_beta05_k - 1) / numel(g1_idx);
    h2 = rgb2hsv(hsv(numel(g2_idx)));
    h2(:,1) = mod(h2(:,1) + hue_offset, 1);
    h2(:,3) = h2(:,3) * 0.75;
    c2 = hsv2rgb(h2);

    % 4 line styles cycle; second cycle adds a marker to distinguish repeats
    beta_styles  = {'-', '--', ':', '-.',  '-', '--', ':', '-.',  '-', '--', ':', '-.'};
    beta_markers = {'none','none','none','none', 'o','s','^','d', 'v','p','h','x'};
    marker_gap   = 400;

    figure('Position', [100 100 1500 420], 'Color', 'w');

    ax1 = subplot(1, 2, 1);
    for k = 1:numel(g1_idx)
        bi    = g1_idx(k);
        beta  = testgroup(bi);
        n_pts = numel(xh_med{mi, bi});
        h_line = semilogy(1:n_pts, xh_med{mi, bi}, ...
            'LineStyle', beta_styles{k}, 'Marker', beta_markers{k}, ...
            'Color', c1(k,:), 'LineWidth', 3, ...
            'DisplayName', sprintf('$\\beta=%.2f$', beta));
        if ~strcmp(beta_markers{k}, 'none')
            h_line.MarkerIndices = 1:marker_gap:n_pts;
        end
        hold on
    end
    % yline(tol, 'k:', 'LineWidth', 3, 'DisplayName', 'Tolerance');
    xlabel('Iteration',Interpreter='latex'); ylabel('$\|x^{(t)} - x_*\|_2$', 'Interpreter', 'latex');
    % title('Coarse \beta search (0.1 – 0.7)');
    ylim([1e-12, 1e2]);
    legend('Location', 'southwest', 'FontSize', 7, 'Interpreter', 'latex');
    grid on; hold off;

    ax2 = subplot(1, 2, 2);
    for k = 1:numel(g2_idx)
        bi    = g2_idx(k);
        beta  = testgroup(bi);
        n_pts = numel(xh_med{mi, bi});
        h_line = semilogy(1:n_pts, xh_med{mi, bi}, ...
            'LineStyle', beta_styles{k}, 'Marker', beta_markers{k}, ...
            'Color', c2(k,:), 'LineWidth', 3, ...
            'DisplayName', sprintf('$\\beta=%.2f$', beta));
        if ~strcmp(beta_markers{k}, 'none')
            h_line.MarkerIndices = 1:marker_gap:n_pts;
        end
        hold on
    end
    % yline(tol, 'k:', 'LineWidth', 3, 'DisplayName', 'Tolerance');
    xlabel('Iteration',Interpreter='latex'); ylabel('$\|x^{(t)} - x_*\|_2$', 'Interpreter', 'latex');
    % title('Fine \beta search (0.50 – 0.60)');
    ylim([1e-12, 1e2]);
    legend('Location', 'southwest', 'FontSize', 7, 'Interpreter', 'latex');
    grid on; hold off;

    % sgtitle(sprintf('%s  —  Beta Search, mag=%g', W_names{mi}, corrupt_mag), ...
        % 'FontSize', 11, 'Interpreter', 'none');
    fontsize(gcf, scale=1.75);

    % Force both axes to the same size (must come AFTER fontsize to stick)
    drawnow;
    ax1.Position(2) = ax2.Position(2);
    ax1.Position(4) = ax2.Position(4);
    set(ax1, 'YMinorTick', 'on');
    set(ax2, 'YMinorTick', 'on');

    exportgraphics(gcf, fullfile(output_dir, ...
        sprintf('betasearch_%s_mag%g.pdf', W_names{mi}, corrupt_mag)), 'ContentType', 'vector');
    savefig(gcf, fullfile(output_dir, ...
        sprintf('betasearch_%s_mag%g.fig', W_names{mi}, corrupt_mag)));
end

fprintf('\nDone. Saved to %s/\n', output_dir);
