% script_gaussian_experiment.m
%
% Multi-seed weighted RK experiment: Standard Gaussian system.
% Matches notebook setup: m=1000, n=50, beta=0.20, mag=100, sys_seed=32, num_seeds=3.
% Compares fChoice1W, fChoice4W, fChoice5W.
%
% Plots produced:
%   1. Approximation error vs iteration (median solid, min/max dashed)
%   2. Contaminated residual norm vs iteration (median solid, min/max dashed)
%   3. Picked row index scatter (green=clean, red=corrupt), median run
%   4. Residual histograms at snapshot iterations, median run

clear; clc;

%% ── Output directory ─────────────────────────────────────────────────────────
% (defined early so it can be used below; also created here)

%% ── Parameters (matching notebook) ──────────────────────────────────────────
m           = 1000;
n           = 50;
beta        =  0.20;
corrupt_mag =  100;
sys_seed    =  6; 
num_seeds   =  3;
t_max       =  10000;
tol         =  1e-6;
eta         =  1.0;
hist_every  =  300;
output_dir  = 'figs_1004';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

%% ── Generate system ──────────────────────────────────────────────────────────
[A, ~, ~, x_true, b_tilde, corrupt_idx] = generate_system( ...
    'corrupt_beta', beta, 'rng_seed', sys_seed, ...
    'n', n, 'm', m, 'corrupt_magnitude', corrupt_mag, ...
    'type', 'normal');
x0 = zeros(size(x_true));
clean_idx = setdiff(1:m, corrupt_idx)';

%% ── Weight functions ─────────────────────────────────────────────────────────
% q = 1 - beta;   % QRK quantile: keep bottom (1-beta) = 0.80 of residuals
q1 = 0.9; q2 = 0.8; q3 = 0.7;
r0     = A * x0 - b_tilde;          % initial residual (x0 = zeros)
thresh0 = median(abs(r0));           % constant threshold


W_funcs  = {@(r,n) qrkW(r, q1, []) ,@(r,n) qrkW(r, q2, []), @(r,n) qrkW(r, q3, []), @(r,n) fConstThreshW(r, thresh0),  @fChoice6W, @fChoice1W, @fChoice4W, @fChoice5W,...
    @fChoice2W, @fChoice7W, @fChoice3W};
W_labels = {sprintf('$\\mathrm{QRK}\\,(q=%.2f)$', q1), sprintf('$\\mathrm{QRK}\\,(q=%.2f)$', q2), sprintf('$\\mathrm{QRK}\\,(q=%.2f)$', q3), sprintf('constant threshold'), ...
            '$e^{-r^2/\mathrm{med}(|r|)}$', ...
            '$e^{-r^2/\mathrm{med}(|r|)^2}$','$e^{-r^2/\sigma_{\mathrm{MAD}}}$', ...
            '$e^{-r^2/\sigma_{\mathrm{MAD}}^2}$', ...
            '$1/|r|$','$1/r^{1.2}$','$1/r^2$'...
            };
W_titles = W_labels;
W_names  = {sprintf('QRK_q%.2f', q1),sprintf('QRK_q%.2f', q2),sprintf('QRK_q%.2f', q3),sprintf('constant_threshold'),'f6','f1', 'f4', 'f5', ...
    'f2','f7', 'f3', ...
    };  % safe filenames
num_methods = numel(W_funcs);

% colors = [0.00 0.45 0.70;   % blue        - f1
%           0.47 0.67 0.19;   % green       - f4
%           0.85 0.33 0.10;   % red-orange  - f5
%           0.50 0.50 0.50;   % gray        - QRK
%           0.93 0.69 0.13;   % yellow      - method 5
%           0.49 0.18 0.56;   % purple      - method 6
%           0.30 0.75 0.93 ...% light blue  - method 7
%           ]; 

% colors = [
%     "QRK": "#f7004e",
%     "exp_med_unsq": "#b95900",
%     "exp_med_sq": "#ff8000",
%     "exp_sig_unsq": "#0950A1E9",
%     "exp_sig_sq": "#3583f9",
%     "inv_unsq": "#00731d",
%     "inv_sq": "#1cc502"];
%     % MATLAB color definitions corresponding to the method labels above


% colors = [247 0 78; 185 89 0; 255 128 0; 9 80 169; ...
%           53 131 249; 0 115 29; 28 197 2] / 255;
colors = [
    0.80 0.10 0.10;   % red - dark
    0.95 0.45 0.45;   % red - mid
    1.00 0.75 0.75;   % red - light
    0.58 0.20 0.75;   % purple
    0.08 0.35 0.75;   % blue - dark
    0.40 0.65 0.92;   % blue - light
    0.90 0.50 0.05;   % orange - dark
    1.00 0.78 0.45;   % orange - light
    0.13 0.55 0.13;   % green - dark
    0.18 0.65 0.25;   % greed - mid
    0.56 0.82 0.56;   % green - light
];
%% ── Run experiments ──────────────────────────────────────────────────────────
xh_med   = cell(num_methods,1);  xh_min   = cell(num_methods,1);  xh_max = cell(num_methods,1);
res_med  = cell(num_methods,1);  res_min  = cell(num_methods,1);  res_max= cell(num_methods,1);
rf_med   = cell(num_methods,1);
sel_med  = cell(num_methods,1);
iters_all= cell(num_methods,1);

for mi = 1:num_methods
    fprintf('Running %s (%s) ...\n', W_labels{mi}, W_titles{mi});
    cand_iters = zeros(1,num_seeds);
    cand = cell(num_seeds,1);

    for si = 1:num_seeds
        rng(41 + si, 'twister');   % seeds 42,43,44,... matching Python algo_seed_start=42
        p_func = @l2erk1p;
        if strcmp(W_labels{mi}, 'QRK'), p_func = @qrkp; end
        [~, resnorm, xh, iter, ~, rf, sel] = wrk(A, b_tilde, x0, x_true, ...
            eta, t_max, tol, W_funcs{mi}, p_func, n, false);
        cand_iters(si) = iter;
        cand{si} = struct('xh',xh,'resnorm',resnorm,'iter',iter,'rf',rf,'sel',sel);
    end

    [~,med_i] = min(abs(cand_iters - median(cand_iters)));
    [~,min_i] = min(cand_iters);
    [~,max_i] = max(cand_iters);

    xh_med{mi}  = cand{med_i}.xh;   xh_min{mi} = cand{min_i}.xh;   xh_max{mi} = cand{max_i}.xh;
    res_med{mi} = cand{med_i}.resnorm; res_min{mi}= cand{min_i}.resnorm; res_max{mi}= cand{max_i}.resnorm;
    rf_med{mi}  = cand{med_i}.rf;
    sel_med{mi} = cand{med_i}.sel;
    iters_all{mi} = cand_iters;

    fprintf('  min=%d  median=%d  max=%d\n', min(cand_iters), round(median(cand_iters)), max(cand_iters));
end

%% ── Plot 1: Approximation error ──────────────────────────────────────────────
%{
figure('Color', 'w');
tl = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tl, sprintf('Gaussian N(0,1), \\beta=%.0f%%, mag=%g — Approximation Error', 100*beta, corrupt_mag));

h = [];
nexttile;
for mi = 1:4
    c = colors(mi,:);
    h(end+1) = semilogy(1:numel(xh_med{mi}), xh_med{mi}, '-',  'Color',c,'LineWidth',2,'DisplayName',W_labels{mi});
    hold on
    % semilogy(1:numel(xh_min{mi}), xh_min{mi}, '--', 'Color',c,'LineWidth',1,'HandleVisibility','off');
    % semilogy(1:numel(xh_max{mi}), xh_max{mi}, '--', 'Color',c,'LineWidth',1,'HandleVisibility','off');
end
h_tol = yline(tol,'k:','LineWidth',1.2,'DisplayName','Tolerance');
xlabel('Iteration'); ylabel('$\|x^{(t)} - x_*\|_2$','Interpreter','latex');
axis square; hold off;

nexttile;
for mi = 5:8
    c = colors(mi,:);
    h(end+1) = semilogy(1:numel(xh_med{mi}), xh_med{mi}, '-',  'Color',c,'LineWidth',2,'DisplayName',W_labels{mi});
    hold on
    % semilogy(1:numel(xh_min{mi}), xh_min{mi}, '--', 'Color',c,'LineWidth',1,'HandleVisibility','off');
    % semilogy(1:numel(xh_max{mi}), xh_max{mi}, '--', 'Color',c,'LineWidth',1,'HandleVisibility','off');
end
yline(tol,'k:','LineWidth',1.2,'HandleVisibility','off');
xlabel('Iteration'); ylabel('$\|x^{(t)} - x_*\|_2$','Interpreter','latex');
axis square; hold off;

nexttile;
for mi = 9:11
    c = colors(mi,:);
    h(end+1) = semilogy(1:numel(xh_med{mi}), xh_med{mi}, '-',  'Color',c,'LineWidth',2,'DisplayName',W_labels{mi});
    hold on
    % semilogy(1:numel(xh_min{mi}), xh_min{mi}, '--', 'Color',c,'LineWidth',1,'HandleVisibility','off');
    % semilogy(1:numel(xh_max{mi}), xh_max{mi}, '--', 'Color',c,'LineWidth',1,'HandleVisibility','off');
end
yline(tol,'k:','LineWidth',1.2,'HandleVisibility','off');
xlabel('Iteration'); ylabel('$\|x^{(t)} - x_*\|_2$','Interpreter','latex');
axis square; hold off;

lg = legend([h, h_tol], 'Interpreter', 'latex', 'FontSize', 8);
lg.Layout.Tile = 'east';
fontsize(gcf, scale=1.5)

exportgraphics(gcf,fullfile(output_dir,sprintf('gaussian_approx_error_mag%g.pdf',corrupt_mag)),'ContentType','vector');
savefig(gcf,fullfile(output_dir,sprintf('gaussian_approx_error_mag%g.fig',corrupt_mag)));
%}
%% ── Plot 2: Contaminated residual norm ───────────────────────────────────────
%{
figure('Position',[100 100 800 420],'Color','w');
for mi = 1:(num_methods-2)
    c = colors(mi,:);
    semilogy(1:numel(res_med{mi}), res_med{mi}, '-',  'Color',c,'LineWidth',2,'DisplayName',W_labels{mi});
    hold on
    semilogy(1:numel(res_min{mi}), res_min{mi}, '--', 'Color',c,'LineWidth',1,'HandleVisibility','off');
    semilogy(1:numel(res_max{mi}), res_max{mi}, '--', 'Color',c,'LineWidth',1,'HandleVisibility','off');
end
xlabel('Iteration'); ylabel('$\|Ax^{(t)} - \tilde{b}\|_2$','Interpreter','latex');
title(sprintf('Gaussian N(0,1), \\beta=%.0f%%, mag=%g — Residual Norm',100*beta,corrupt_mag));
legend('Location','northeast','FontSize',9,'Interpreter','latex'); grid on; hold off;
fontsize(gcf, scale=1.5)

exportgraphics(gcf,fullfile(output_dir,sprintf('gaussian_residual_norm_mag%g.pdf',corrupt_mag)),'ContentType','vector');
savefig(gcf,fullfile(output_dir,sprintf('gaussian_residual_norm_mag%g.fig',corrupt_mag)));
%}
%% ── Plot 3: Picked row index scatter (median run) ────────────────────────────
%{
for mi = 8:9
    sel  = sel_med{mi};
    iters= 1:numel(sel);
    is_corrupt = ismember(sel, corrupt_idx);

    figure('Position',[100 100 800 320],'Color','w'); hold on;
    h1 = plot(iters(~is_corrupt), sel(~is_corrupt), 'o', ...
        'MarkerSize',3,'Color',[0.13 0.55 0.13],'MarkerFaceColor',[0.13 0.55 0.13], ...
        'LineStyle','none','DisplayName','Clean row');
    h2 = plot(iters( is_corrupt), sel( is_corrupt), 'x', ...
        'MarkerSize',5,'Color',[0.85 0.33 0.10],'LineWidth',1.0,'LineStyle','none', ...
        'DisplayName','Corrupt row');
    xlabel('Iteration'); ylabel('Picked row index');
    title(sprintf('Gaussian N(0,1) — Picked row index: %s', W_names{mi}));
    legend('Location','northeast','FontSize',9,'Interpreter','latex'); grid on; hold off;
    exportgraphics(gcf,fullfile(output_dir,sprintf('gaussian_pickedrow_%s_mag%g.pdf',W_names{mi},corrupt_mag)),'ContentType','vector');
savefig(gcf,fullfile(output_dir,sprintf('gaussian_pickedrow_%s_mag%g.fig',W_names{mi},corrupt_mag)));
end
fontsize(gcf, scale=1.5)
%}
%% ── Plot 4: Residual histograms at snapshots (first/last run) ────────────────────
%{
for mi = 1:11
    rf   = rf_med{mi};
    T    = size(rf,2);
    snaps= [1 T];
    if isempty(snaps), snaps = T; end
    if numel(snaps) > 6
        idx6  = round(linspace(1, numel(snaps), 2));
        snaps = snaps(idx6);
    end

    ncols = min(3, numel(snaps));
    nrows = ceil(numel(snaps)/ncols);
    all_vals = abs(rf(:));
    bin_edges = linspace(0, max(all_vals)*1.02, 51);

    figure('Position',[100 100 420*ncols 300*nrows],'Color','w');
    for fi = 1:numel(snaps)
        subplot(nrows,ncols,fi);
        r_snap = rf(:, snaps(fi));
        histogram(abs(r_snap(clean_idx)),   bin_edges,'FaceColor',[0.13 0.55 0.13],'FaceAlpha',0.55,'DisplayName','Clean');
        hold on;
        histogram(abs(r_snap(corrupt_idx)), bin_edges,'FaceColor',[0.85 0.33 0.10],'FaceAlpha',0.55,'DisplayName','Corrupt');
        xlabel('|r_i|'); ylabel('Count');
        title(sprintf('Iter %d', snaps(fi)),'FontSize',9);
        legend('FontSize',7,'Location','northeast');
        grid on; hold off;
    end
    sgtitle(sprintf('Gaussian N(0,1) — Residual Histograms: %s', W_names{mi}),'FontSize',10);
    exportgraphics(gcf,fullfile(output_dir,sprintf('gaussian_histograms_%s_mag%g.pdf',W_names{mi},corrupt_mag)),'ContentType','vector');
savefig(gcf,fullfile(output_dir,sprintf('gaussian_histograms_%s_mag%g.fig',W_names{mi},corrupt_mag)));
end
fontsize(gcf, scale=1.5)
%}
%% ── Plot 5: Weights evaluated on residuals at snapshots (median run) ─────────
%{
sort_residuals = true;   % true  → sort rows by |r| before plotting (x-axis = |r|)
                         % false → keep original row order (x-axis = row index)

for mi = 8:9
    rf  = rf_med{mi};
    T   = size(rf, 2);
    snaps = 1:hist_every:T;
    if isempty(snaps), snaps = T; end
    if numel(snaps) > 3
        idx6  = round(linspace(1, numel(snaps), 3));
        snaps = snaps(idx6);
    end

    ncols = min(3, numel(snaps));
    nrows = ceil(numel(snaps) / ncols);

    is_corrupt_row             = false(m, 1);
    is_corrupt_row(corrupt_idx) = true;

    figure('Position', [100 100 420*ncols 300*nrows], 'Color', 'w');
    for fi = 1:numel(snaps)
        subplot(nrows, ncols, fi);

        r_snap = rf(:, snaps(fi));
        Wmat   = W_funcs{mi}(r_snap, n);
        w_vec  = diag(Wmat);

        if sort_residuals
            [~, sort_idx] = sort(abs(r_snap));
            w_plot       = w_vec(sort_idx);
            corrupt_plot = is_corrupt_row(sort_idx);
            x_vals       = (1:m)';
            xlab         = 'Rank (sorted by |r_i|)';
        else
            x_vals       = (1:m)';
            w_plot       = w_vec;
            corrupt_plot = is_corrupt_row;
            xlab         = 'Row index';
        end

        plot(x_vals(~corrupt_plot), w_plot(~corrupt_plot), '.', ...
            'Color', [0.13 0.55 0.13], 'MarkerSize', 4, 'DisplayName', 'Clean');
        hold on;
        plot(x_vals(corrupt_plot), w_plot(corrupt_plot), 'x', ...
            'Color', [0.85 0.33 0.10], 'MarkerSize', 6, 'LineWidth', 1.2, ...
            'DisplayName', 'Corrupt');
        xlabel(xlab); ylabel('Weight');
        title(sprintf('Iter %d', snaps(fi)), 'FontSize', 9);
        legend('FontSize', 7, 'Location', 'northeast');
        grid on; hold off;
    end
    sgtitle(sprintf('Gaussian N(0,1) — Weights at Snapshots: %s', W_names{mi}), 'FontSize', 10);
    exportgraphics(gcf, fullfile(output_dir, sprintf('gaussian_weights_%s_mag%g.pdf', W_names{mi}, corrupt_mag)), 'ContentType', 'vector');
    savefig(gcf, fullfile(output_dir, sprintf('gaussian_weights_%s_mag%g.fig', W_names{mi}, corrupt_mag)));
end

%% ── Summary table ────────────────────────────────────────────────────────────
fprintf('\n%-6s | %8s | %10s | %8s\n','Method','Min','Meddian','Max');
fprintf('%s\n',repmat('-',1,40));
for mi = 1:num_methods
    ic = iters_all{mi};
    fprintf('%-6s | %8d | %10d | %8d\n',W_labels{mi},min(ic),round(median(ic)),max(ic));
end
fprintf('%s\n',repmat('-',1,40));
fprintf('Saved PDFs to %s/\n',output_dir);
%}