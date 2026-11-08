%% Organized Excel -> Fx/Fy/Fz RMSE/MAE, optional contact-ID figures
% MATLAB R2020b+. No additional toolboxes. Does not read Python code, models, or the current YAML.
% Fc=sqrt(Fx^2+Fy^2) in the previously supplied code is not Fz.
% Stored ground truth is the Kalman-filtered, aligned force. Preserve filtering, zero, sign, and axis order.
% AUTHOR: Daeyun
clear; clc;

%% 1) Settings: an empty path opens a file selection dialog at runtime.
force_file = "";  % force-estimation_class0_...xlsx or a loss-mask file
force_source = "";                    % Default: all recordings in the selected Excel file. Set a CSV filename to filter.
force_load = "all";                    % "all", "loaded", "unloaded"
make_id_plots = false;                 % Set true to also plot contact-ID results.
id_file = "";                         % id-estimation_all-models_...xlsx
id_model = "transformer";              % Same model column to compare across both modes.
id_range = 1:18;                       % Default: all IDs, including the tip. Use 1:17 for body only.
tolerances = 0:5;                      % 0 = exact ID. Both +/-2 and +/-3 are computed here.

%% 2) Force: discover model columns and use rows valid across XYZ for each model.
if strlength(force_file) == 0
    [name, folder] = uigetfile('*.xlsx', 'Select an organized force Excel file');
    if isequal(name, 0), return; end
    force_file = string(fullfile(folder, name));
end
T = readtable(force_file, 'Sheet', 'force_all', 'VariableNamingRule', 'preserve');
R = readtable(force_file, 'Sheet', '_rows', 'VariableNamingRule', 'preserve');
assert(height(T) == height(R), 'force_all and _rows have different row counts.');
keep = true(height(T), 1);
if strlength(force_source) > 0
    % Support both Linux and Windows source paths. Preserve row order and measurements.
    basename = regexprep(string(R.source_file), '^.*[\/]', '');
    keep = keep & basename == force_source;
end
assert(ismember(force_load, ["all", "loaded", "unloaded"]), 'Invalid force_load setting');
if force_load ~= "all"
    keep = keep & string(R.load_state_gt) == force_load;
end
assert(any(keep), 'No rows match the selected source file and load-state conditions.');
vars = string(T.Properties.VariableNames);
fx_cols = vars(startsWith(vars, "pred_fx_") & endsWith(vars, "_mN"));
models = regexprep(fx_cols, '^pred_fx_(.+)_mN$', '$1');
assert(~isempty(models), 'The organized force_all sheet must contain pred_fx_MODEL_mN columns.');
gt = numeric_columns(T, ["gt_fx_mN", "gt_fy_mN", "gt_fz_mN"]);
force_metrics = table();
for model_index = 1:numel(models)
    m = models(model_index);
    pred = numeric_columns(T, "pred_" + ["fx", "fy", "fz"] + "_" + m + "_mN");
    valid = keep & all(~isnan([gt, pred]) & ~isinf([gt, pred]), 2);
    err = pred(valid, :) - gt(valid, :);
    rmse = sqrt(mean(err.^2, 1));
    mae = mean(abs(err), 1);
    % XYZ: average over all three axis components, not the RMSE of the vector error norm.
    row = table(m, sum(valid), rmse(1), mae(1), rmse(2), mae(2), rmse(3), mae(3), ...
        sqrt(mean(err(:).^2)), mean(abs(err(:))), 'VariableNames', ...
        {'Model','N','RMSE_fx_mN','MAE_fx_mN','RMSE_fy_mN','MAE_fy_mN', ...
         'RMSE_fz_mN','MAE_fz_mN','RMSE_xyz_mN','MAE_xyz_mN'});
    force_metrics = [force_metrics; row]; %#ok<AGROW>
end
disp(force_metrics);
% Different N values across models mean the evaluation rows are not identical.
[folder, ~, ~] = fileparts(force_file);
output_file = fullfile(folder, "matlab_metrics_" + string(datetime('now', ...
    'TimeZone','Asia/Seoul','Format','yyyyMMdd_HHmmss_SSS')) + ".xlsx");
assert(~isfile(output_file), 'A result file with the same name already exists.');
writetable(force_metrics, output_file, 'Sheet', 'force_metrics');
settings = table(force_file, force_source, force_load);
writetable(settings, output_file, 'Sheet', 'force_settings');
fprintf('Saved: %s\n', output_file);

%% 3) ID: recompute +/-3 from the original predictions even if organize used id-tol 2.
if make_id_plots
    if strlength(id_file) == 0
        [name, folder] = uigetfile('*.xlsx', 'Select an organized ID Excel file');
        if isequal(name, 0), return; end
        id_file = string(fullfile(folder, name));
    end
    M = readtable(id_file, 'Sheet', 'loss-mask_id', 'VariableNamingRule','preserve');
    C = readtable(id_file, 'Sheet', 'class0_id_conditional', 'VariableNamingRule','preserve');
    F = readtable(id_file, 'Sheet', 'class0_id', 'VariableNamingRule','preserve');
    assert(isequaln(M.gt_id,C.gt_id,F.gt_id) && ...
        isequaln(M.loaded_gt,C.loaded_gt,F.loaded_gt), 'Ground truth or row alignment differs across ID sheets.');
    g = numeric_columns(M, "gt_id");
    pm = numeric_columns(M, id_model);
    pc = numeric_columns(C, id_model);
    pf = numeric_columns(F, id_model);
    valid = M.loaded_gt == 1 & ismember(g, id_range) & ...
        all(~isnan([g,pm,pc,pf]) & ~isinf([g,pm,pc,pf]), 2);
    g=g(valid); pm=pm(valid); pc=pc(valid); pf=pf(valid);
    assert(~isempty(g), 'No ground-truth-loaded evaluation rows exist in the selected ID range.');
    dm=abs(pm-g); dc=abs(pc-g);
    % The denominator includes all ground-truth-loaded rows, including Class0 predictions of ID0.
    tol = tolerances(:);
    masked_pct = arrayfun(@(k) 100*mean(dm<=k), tol);
    class0_pct = arrayfun(@(k) 100*mean(dc<=k), tol);
    detected_pct = arrayfun(@(k) 100*mean((pf>0) & abs(pf-g)<=k), tol);
    id_accuracy = table(tol, repmat(numel(g),size(tol)), masked_pct, class0_pct, detected_pct, ...
        'VariableNames', {'Tolerance','N','Masked_conditional_pct','Class0_conditional_pct','Class0_detected_pct'});
    disp(id_accuracy);
    writetable(id_accuracy, output_file, 'Sheet', 'id_tolerance');
    writetable(table(id_file,id_model,min(id_range),max(id_range), ...
        'VariableNames',{'File','Model','MinID','MaxID'}), output_file, 'Sheet', 'id_settings');

    % A: rows = ground-truth IDs; columns = predicted IDs. Supported rows sum to 100%; empty rows are gray.
    figure('Name','ID confusion','Position',[100 100 1200 500]);
    tiledlayout(1,2,'TileSpacing','compact');
    nexttile; plot_confusion(g,pm,id_range,1:18,'Masked: conditional ID');
    nexttile; plot_confusion(g,pf,id_range,0:18,'Class0: final ID (0 = unloaded)');

    % B: show the fraction of correctly predicted rows as the allowed ID distance increases.
    figure('Name','ID tolerance accuracy');
    plot(tol,[masked_pct,class0_pct,detected_pct],'-o','LineWidth',1.5);
    grid on; ylim([0 100]); xticks(tol);
    xlabel('Allowed ID distance n (0 = exact ID)'); ylabel('Accuracy (%)');
    title(sprintf('%s: GT loaded ID%d-%d, N=%d',id_model,min(id_range),max(id_range),numel(g)), ...
        'Interpreter','none');
    legend('Masked: conditional','Class0: conditional','Class0: including detection','Location','best');

    % C: per-ID +/-2 performance. IDs with no samples remain NaN rather than appearing as 0%.
    ids=id_range(:); counts=zeros(size(ids)); scores=nan(numel(ids),3);
    for i=1:numel(ids)
        selected=g==ids(i); counts(i)=sum(selected);
        if counts(i)>0
            scores(i,:)=100*[mean(dm(selected)<=2),mean(dc(selected)<=2), ...
                mean((pf(selected)>0) & abs(pf(selected)-g(selected))<=2)];
        end
    end
    per_id=table(ids,counts,scores(:,1),scores(:,2),scores(:,3),'VariableNames', ...
        {'GT_ID','N','Masked_within2_pct','Class0_within2_pct','Class0_detected_within2_pct'});
    writetable(per_id,output_file,'Sheet','id_per_segment');
    figure('Name','Per-ID recall','Position',[100 100 1200 450]);
    bar(ids,scores,'grouped'); grid on; ylim([0 100]); xticks(ids);
    xlabel('Ground-truth contact ID'); ylabel('Within +/-2 recall (%)');
    legend('Masked: conditional','Class0: conditional','Class0: including detection','Location','best');
end

%% Helper functions
function values = numeric_columns(T, names)
    values = nan(height(T),numel(names));
    for k=1:numel(names)
        v=T.(char(names(k)));
        if isnumeric(v), values(:,k)=double(v);
        else, values(:,k)=str2double(string(v)); end
    end
end

function plot_confusion(gt,pred,ids,pred_ids,label)
    counts=accumarray([gt,pred+1],1,[18,19]);
    counts=counts(ids,pred_ids+1);
    percentages=100*counts./sum(counts,2);
    h=imagesc(pred_ids,ids,percentages);
    set(h,'AlphaData',~isnan(percentages)); set(gca,'Color',[.85 .85 .85]);
    caxis([0 100]); colorbar; xticks(pred_ids); yticks(ids);
    xlabel('Predicted ID'); ylabel('Ground-truth ID'); title(label);
    hold on;
    plot(ids,ids,'w-'); plot(ids-2,ids,'w:'); plot(ids+2,ids,'w:');
    xlim([min(pred_ids)-.5,max(pred_ids)+.5]); ylim([min(ids)-.5,max(ids)+.5]);
end
