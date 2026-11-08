%% Organized Excel -> Fx/Fy/Fz RMSE/MAE, optional contact-ID figures
% MATLAB R2020b+. 추가 toolbox 불필요. Python/모델/현재 YAML을 읽지 않습니다.
% 기존 첨부 코드의 Fc=sqrt(Fx^2+Fy^2)는 Fz가 아닙니다.
% 저장 GT는 Kalman aligned 힘입니다. 필터/영점/부호/축 순서를 바꾸지 않습니다.
clear; clc;

%% 1) 설정: 빈 경로는 실행 시 파일 선택 창을 엽니다.
force_file = "";  % force-estimation_class0_...xlsx 또는 loss-mask 파일
force_source = "";                    % 기본: 선택 Excel의 모든 실험. 특정 CSV만 보려면 파일명 지정
force_load = "all";                    % "all", "loaded", "unloaded"
make_id_plots = false;                 % ID 그래프도 그리려면 true
id_file = "";                         % id-estimation_all-models_...xlsx
id_model = "transformer";              % 두 방식에서 비교할 동일 모델 열 이름
id_range = 1:18;                       % 기본: tip 포함 전체 ID. body만 보려면 1:17
tolerances = 0:5;                      % 0=정확ID. ±2/±3 모두 여기서 계산

%% 2) 힘: 모델 열을 자동 탐색하고 각 모델의 XYZ 공통 유효 행으로 계산
if strlength(force_file) == 0
    [name, folder] = uigetfile('*.xlsx', 'Organized force Excel 선택');
    if isequal(name, 0), return; end
    force_file = string(fullfile(folder, name));
end
T = readtable(force_file, 'Sheet', 'force_all', 'VariableNamingRule', 'preserve');
R = readtable(force_file, 'Sheet', '_rows', 'VariableNamingRule', 'preserve');
assert(height(T) == height(R), 'force_all과 _rows의 행 수가 다릅니다.');
keep = true(height(T), 1);
if strlength(force_source) > 0
    % Linux/Windows 원본 경로 모두 지원. 행 순서/실측값은 그대로 유지합니다.
    basename = regexprep(string(R.source_file), '^.*[\/]', '');
    keep = keep & basename == force_source;
end
assert(ismember(force_load, ["all", "loaded", "unloaded"]), 'force_load 설정 오류');
if force_load ~= "all"
    keep = keep & string(R.load_state_gt) == force_load;
end
assert(any(keep), '선택한 원본 파일/유무부하 조건에 해당하는 행이 없습니다.');
vars = string(T.Properties.VariableNames);
fx_cols = vars(startsWith(vars, "pred_fx_") & endsWith(vars, "_mN"));
models = regexprep(fx_cols, '^pred_fx_(.+)_mN$', '$1');
assert(~isempty(models), 'Organized force_all의 pred_fx_MODEL_mN 열이 필요합니다.');
gt = numeric_columns(T, ["gt_fx_mN", "gt_fy_mN", "gt_fz_mN"]);
force_metrics = table();
for model_index = 1:numel(models)
    m = models(model_index);
    pred = numeric_columns(T, "pred_" + ["fx", "fy", "fz"] + "_" + m + "_mN");
    valid = keep & all(~isnan([gt, pred]) & ~isinf([gt, pred]), 2);
    err = pred(valid, :) - gt(valid, :);
    rmse = sqrt(mean(err.^2, 1));
    mae = mean(abs(err), 1);
    % XYZ: 3축 성분 전체의 평균. 벡터 오차 norm의 RMSE와 구분합니다.
    row = table(m, sum(valid), rmse(1), mae(1), rmse(2), mae(2), rmse(3), mae(3), ...
        sqrt(mean(err(:).^2)), mean(abs(err(:))), 'VariableNames', ...
        {'Model','N','RMSE_fx_mN','MAE_fx_mN','RMSE_fy_mN','MAE_fy_mN', ...
         'RMSE_fz_mN','MAE_fz_mN','RMSE_xyz_mN','MAE_xyz_mN'});
    force_metrics = [force_metrics; row]; %#ok<AGROW>
end
disp(force_metrics);
% 모델별 N이 다르면 같은 평가 행의 비교가 아님에 주의합니다.
[folder, ~, ~] = fileparts(force_file);
output_file = fullfile(folder, "matlab_metrics_" + string(datetime('now', ...
    'TimeZone','Asia/Seoul','Format','yyyyMMdd_HHmmss_SSS')) + ".xlsx");
assert(~isfile(output_file), '동일 이름의 결과 파일이 이미 있습니다.');
writetable(force_metrics, output_file, 'Sheet', 'force_metrics');
settings = table(force_file, force_source, force_load);
writetable(settings, output_file, 'Sheet', 'force_settings');
fprintf('Saved: %s\n', output_file);

%% 3) ID: organize 시 id-tol이 2였어도 원 예측에서 ±3을 다시 계산할 수 있습니다.
if make_id_plots
    if strlength(id_file) == 0
        [name, folder] = uigetfile('*.xlsx', 'Organized ID Excel 선택');
        if isequal(name, 0), return; end
        id_file = string(fullfile(folder, name));
    end
    M = readtable(id_file, 'Sheet', 'loss-mask_id', 'VariableNamingRule','preserve');
    C = readtable(id_file, 'Sheet', 'class0_id_conditional', 'VariableNamingRule','preserve');
    F = readtable(id_file, 'Sheet', 'class0_id', 'VariableNamingRule','preserve');
    assert(isequaln(M.gt_id,C.gt_id,F.gt_id) && ...
        isequaln(M.loaded_gt,C.loaded_gt,F.loaded_gt), 'ID 시트 간 GT/행 정렬이 다릅니다.');
    g = numeric_columns(M, "gt_id");
    pm = numeric_columns(M, id_model);
    pc = numeric_columns(C, id_model);
    pf = numeric_columns(F, id_model);
    valid = M.loaded_gt == 1 & ismember(g, id_range) & ...
        all(~isnan([g,pm,pc,pf]) & ~isinf([g,pm,pc,pf]), 2);
    g=g(valid); pm=pm(valid); pc=pc(valid); pf=pf(valid);
    assert(~isempty(g), '선택한 ID 범위의 GT 유부하 평가 행이 없습니다.');
    dm=abs(pm-g); dc=abs(pc-g);
    % 분모는 언제나 GT 유부하 전체. class0의 ID0 예측을 분모에서 빼지 않습니다.
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

    % A: 행=정답ID, 열=예측ID. 각 GT ID 행의 합이100%, 무표본 행은 회색입니다.
    figure('Name','ID confusion','Position',[100 100 1200 500]);
    tiledlayout(1,2,'TileSpacing','compact');
    nexttile; plot_confusion(g,pm,id_range,1:18,'Masked: conditional ID');
    nexttile; plot_confusion(g,pf,id_range,0:18,'Class0: final ID (0 = unloaded)');

    % B: 허용 오차를 늘릴 때 실제로 얼마나 많은 행을 맞히는지 보여줍니다.
    figure('Name','ID tolerance accuracy');
    plot(tol,[masked_pct,class0_pct,detected_pct],'-o','LineWidth',1.5);
    grid on; ylim([0 100]); xticks(tol);
    xlabel('Allowed ID distance n (0 = exact ID)'); ylabel('Accuracy (%)');
    title(sprintf('%s: GT loaded ID%d-%d, N=%d',id_model,min(id_range),max(id_range),numel(g)), ...
        'Interpreter','none');
    legend('Masked: conditional','Class0: conditional','Class0: including detection','Location','best');

    % C: ID별 ±2 성능. 표본이 없는 ID는 NaN이며 0%로 표시하지 않습니다.
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

%% 보조 함수
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
