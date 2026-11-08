%% analysis.py의 비교 Excel 하나 -> ID 분석·그림 전용
% MATLAB R2020b+, 추가 toolbox 없음. 이 파일 상단의 설정만 바꾸면 됩니다.
% 원본 CSV / Python / 모델 / YAML / 다른 Excel은 읽지 않습니다.
% 저장된 최종 ID 정확도와 부하 판별 지표를 그립니다. 새 학습이나 GT 재라벨링 없음.
% 혼동행렬: GT 유부하의 실제 ID별 예측 분포. softmax 확률이 아닙니다.
% Class0의 예측 ID0은 모든 +/-n에서 실패이며, 표본 없는 ID는 회색/빈칸입니다.
clear; clc;

%% 1) 사용자 설정
excel_file = "";          % 빈칸: 실행하면 force_and_id_all_combinations.xlsx 선택
output_folder = "";       % 빈칸: 선택 Excel 옆 matlab_id_날짜시간 폴더 생성
input_filter = strings(0,1); % 모두: 그대로. 예: ["cable_loadcell_angle","cable_loadcell_fk_tip_tf"]
run_filter = strings(0,1); % 모두: 그대로. 예: "20261014_cable_loadcell_angle" (폴더명 또는 전체 경로)
id_group = "all";         % 기본: tip 포함 전체 ID. "body", "tip" 선택 가능 (GT 유부하만 평가)
tolerances = [];          % []: Excel에 있는 값 전부. 예: [0 2 3]
make_confusions = true;   % 입력별 Masked/Class0 혼동행렬
make_id_by_id = true;     % 입력별 ID 정확/허용거리 성공률과 평가행 수
save_fig = true;          % MATLAB에서 다시 편집할 수 있는 .fig도 저장

%% 2) 이 Excel에서 필요한 집계 시트만 읽기
if strlength(excel_file) == 0
    [file, folder] = uigetfile('*.xlsx', 'analysis 비교 Excel 하나 선택');
    if isequal(file, 0), return; end
    excel_file = string(fullfile(folder, file));
end
assert(isfile(excel_file), 'Excel 파일 경로를 확인하세요.');
assert(ismember(id_group, ["all","body","tip"]), 'id_group 설정 오류');
sheets = string(sheetnames(excel_file));
assert(ismember("id_selected", sheets), ...
    'analysis.py가 만든 비교 Excel을 선택하세요. organized의 행별 Excel과 다릅니다.');
I = select_runs(read_sheet(excel_file, "id_selected"), input_filter, run_filter);
Isel = I(string(I.group)==id_group, :);
assert(~isempty(Isel), '선택한 input/run/id_group의 집계행이 없습니다.');
if strlength(output_folder) == 0
    output_folder = fullfile(fileparts(excel_file), "matlab_id_" + ...
        string(datetime('now','Format','yyyyMMdd_HHmmss_SSS')));
end
assert(~isfolder(output_folder), '기존 결과를 보존합니다. 새 output_folder를 지정하세요.');
mkdir(output_folder);

% Python의 ±n 채점 결과를 그대로 사용합니다. 저장되지 않은 n은 Python에서 먼저 계산.
names = string(Isel.Properties.VariableNames);
cols = names(~cellfun('isempty', regexp(cellstr(names), '^accuracy_tol\d+_pct$', 'once')));
available = str2double(regexprep(cols, '^accuracy_tol(\d+)_pct$', '$1'));
[available, order] = sort(available); cols = cols(order);
assert(~isempty(available), 'Excel에 ID 허용거리 정확도 열이 없습니다.');
default_tolerances = isempty(tolerances);
if default_tolerances, tolerances = available; end
tolerances = unique(tolerances(:)');
assert(all(ismember(tolerances, available)), ...
    '요청한 tolerance가 Excel에 없습니다. analysis.py --id-tolerances 옵션으로 먼저 생성하세요.');
[~, order] = ismember(tolerances, available); cols = cols(order);

%% 3) ID 요약: 허용거리별 위치 정확도와 Class0 부하 판별
fig = figure('Color','w','Position',[50 50 1500 650]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile; hold on;
runs = unique(string(Isel.run), 'stable'); colors = lines(numel(runs));
for k = 1:height(Isel)
    style = '-'; if string(Isel.mode(k))=="class0", style='--'; end
    color_index = find(runs==string(Isel.run(k)),1);
    plot(tolerances, numbers(Isel(k,:), cols), [style 'o'], ...
        'Color',colors(color_index,:),'LineWidth',1.5, ...
        'DisplayName',char(model_labels(Isel(k,:))));
end
xticks(tolerances); ylim([0 105]); grid on;
xlabel('Allowed segment error (+/-n)'); ylabel('Accuracy [%]');
title(id_group + " ID / final output / GT-loaded rows");
legend('Location','southoutside','NumColumns',2,'Interpreter','none','FontSize',8);
nexttile;
D = Isel(string(Isel.mode)=="class0",:);
if ~isempty(D)
    bar(numbers(D,["detection_loaded_recall_pct","detection_unloaded_recall_pct"]),'grouped');
    xticks(1:height(D));
    xticklabels(input_labels(D.input) + " / loaded N=" + string(D.n_loaded) + ...
        " / unloaded N=" + string(D.detection_n_unloaded));
    xtickangle(30); set(gca,'TickLabelInterpreter','none','FontSize',8);
    legend('Loaded recall','Unloaded recall','Location','best');
else
    text(.5,.5,'Masked has no load detector','HorizontalAlignment','center');
end
ylim([0 105]); ylabel('Recall [%]'); grid on;
title("Class0 / " + id_group + "-file load detection");
sgtitle('Validation-selected ID models; Class0 ID0 fails every tolerance; accuracy is not softmax probability');
save_plot(fig, output_folder, "id_summary_" + id_group, save_fig);

%% 4) 입력별 ID 혼동행렬: 실제 유부하 ID1-18, Class0에는 예측0 열 포함
% 이 그림은 id_group과 관계없이 ID1-18 전체를 표시하여 Python PNG와 대응합니다.
if make_confusions
    C = select_runs(read_sheet(excel_file,"confusion"), input_filter, run_filter);
    runs = unique(string(C.run),'stable');
    for k = 1:numel(runs)
        R = C(string(C.run)==runs(k),:); modes = unique(string(R.mode),'stable');
        fig = figure('Color','w','Position',[70 70 800*numel(modes) 750]);
        tiledlayout(1,numel(modes),'TileSpacing','compact','Padding','compact');
        for m = 1:numel(modes)
            nexttile; part = R(string(R.mode)==modes(m) & numbers(R,"gt_id")>0,:);
            gt = unique(numbers(part,"gt_id")); pred = unique(numbers(part,"predicted_id"));
            [~,ri] = ismember(numbers(part,"gt_id"),gt);
            [~,ci] = ismember(numbers(part,"predicted_id"),pred);
            counts = accumarray([ri,ci],numbers(part,"count"),[numel(gt),numel(pred)]);
            support = sum(counts,2); percent = nan(size(counts));
            percent(support>0,:) = 100*counts(support>0,:)./support(support>0);
            im = imagesc(percent,[0 100]); set(im,'AlphaData',~isnan(percent));
            set(gca,'Color',[.85 .85 .85]); colormap(parula); cb=colorbar; cb.Label.String='Within-GT-ID percentage [%]';
            xticks(1:numel(pred)); xticklabels(pred);
            yticks(1:numel(gt)); yticklabels(string(gt)+" (N="+string(support)+")");
            xlabel('Predicted ID (0 = unloaded)'); ylabel('GT contact ID on loaded rows');
            title(input_labels(part.input(1))+" / "+modes(m)+newline+ ...
                "Force "+string(part.force_model(1))+" + ID "+string(part.location_model(1)), 'Interpreter','none');
        end
        sgtitle('GT-loaded rows / row-normalized prediction counts / gray = no evaluation rows');
        save_plot(fig,output_folder,"id_confusion_"+string(sprintf('%02d',k))+"_"+safe_name(R.input(1)),save_fig);
    end
end

%% 5) ID별 성공률: validation으로 선택한 ID 모델만, 대표분기 전체와 섞지 않음
if make_id_by_id
    B = select_runs(read_sheet(excel_file,"id_by_id"), input_filter, run_filter);
    B = B(numbers(B,"selected_id")==1,:);
    ids = 1:18;
    if id_group=="body", ids=1:17; elseif id_group=="tip", ids=18; end
    % 기본 그림은 정확 ID와 +/-2, +/-3. 다른 n은 상단 tolerances에서 지정.
    shown = tolerances;
    if default_tolerances, shown=tolerances(ismember(tolerances,[0 2 3])); end
    if isempty(shown), shown=tolerances; end
    shown_cols = "accuracy_tol"+string(shown)+"_pct";
    legends = "+/-"+string(shown)+" segments"; legends(shown==0)="Exact ID";
    runs = unique(string(Isel.run),'stable');
    for k=1:numel(runs)
        R = B(string(B.run)==runs(k),:);
        modes = unique(string(R.mode),'stable');
        assert(~isempty(modes), 'id_by_id에 선택 모델의 행이 없습니다.');
        fig=figure('Color','w','Position',[70 70 800*numel(modes) 850]);
        tiledlayout(2,numel(modes),'TileSpacing','compact','Padding','compact');
        for m=1:numel(modes)
            part=R(string(R.mode)==modes(m) & ismember(numbers(R,"gt_id"),ids),:);
            assert(~isempty(part), 'id_by_id에 선택한 group의 ID가 없습니다.');
            assert(height(part)==numel(unique(numbers(part,"gt_id"))), ...
                'id_by_id에 같은 ID의 선택 모델이 중복되어 있습니다.');
            values=nan(numel(ids),numel(shown)); support=zeros(numel(ids),1);
            [~,at]=ismember(numbers(part,"gt_id"),ids);
            values(at,:)=numbers(part,shown_cols); support(at)=numbers(part,"n_loaded");
            values(support==0,:)=NaN; % N=0을 0%로 표시하지 않음
            nexttile(m); bar(1:numel(ids),values,'grouped'); hold on;
            for missing=find(support==0)'
                patch(missing+[-.45 .45 .45 -.45],[0 0 100 100],[.85 .85 .85], ...
                    'EdgeColor','none','HandleVisibility','off');
            end
            xticks(1:numel(ids)); xticklabels(ids); xlim([.5 numel(ids)+.5]);
            ylim([0 105]); ylabel('Accuracy [%]'); xlabel('GT contact ID'); grid on;
            legend(legends,'Location','southoutside','NumColumns',numel(shown));
            title(input_labels(R.input(1))+" / "+modes(m)+" / "+id_group+newline+ ...
                "Force "+string(part.force_model(1))+" + ID "+string(part.location_model(1)), 'Interpreter','none');
            nexttile(numel(modes)+m); bar(1:numel(ids),support,'FaceColor',[.4 .5 .65]);
            xticks(1:numel(ids)); xticklabels(ids); xlim([.5 numel(ids)+.5]);
            xlabel('GT contact ID'); ylabel('GT-loaded evaluation rows (N)'); grid on;
            title("Total N="+string(sum(support))+"; N=0 IDs have no accuracy estimate");
        end
        sgtitle('Final ID output; Class0 ID0 counts as failure; gray = no evaluation rows');
        save_plot(fig,output_folder,"id_by_id_"+string(sprintf('%02d',k))+"_"+safe_name(R.input(1))+"_"+id_group,save_fig);
    end
end
fprintf('그림 저장: %s\n',output_folder);

%% 아래는 공통 그리기 함수입니다.
function T = read_sheet(file, sheet)
T = readtable(file,'Sheet',sheet,'VariableNamingRule','preserve','TextType','string');
end

function T = select_runs(T, inputs, runs)
% 입력 이름과 실행 폴더 이름만 비교하며, 파일 시스템을 조회하지 않습니다.
inputs=string(inputs); inputs=inputs(strlength(inputs)>0);
runs=string(runs); runs=runs(strlength(runs)>0);
if ~isempty(inputs), T=T(ismember(string(T.input),inputs),:); end
if ~isempty(runs)
    paths=string(T.run);
    folders=regexprep(paths,'^.*[\\/]','');
    T=T(ismember(paths,runs) | ismember(folders,runs),:);
end
end

function name = safe_name(value)
name=regexprep(string(value),'[^A-Za-z0-9_+.-]','_');
end

function values = numbers(T, columns)
columns = string(columns); values = nan(height(T),numel(columns));
for k=1:numel(columns)
    v = T.(char(columns(k)));
    if isnumeric(v) || islogical(v)
        values(:,k)=double(v);
    else
        s = string(v); s(strcmpi(s,"true"))="1"; s(strcmpi(s,"false"))="0";
        values(:,k)=str2double(s);
    end
end
end

function labels = input_labels(values)
labels=string(values);
names=["cable_loadcell","cable_angle","loadcell_angle","cable_loadcell_angle", ...
    "cable_loadcell_estimated_pose","cable_loadcell_fk_tip_tf"];
short=["C+L","C+A","L+A","C+L+A","C+L+Estimated","C+L+FK TF"];
for k=1:numel(names), labels(labels==names(k))=short(k); end
end

function labels = model_labels(T)
labels=input_labels(T.input)+" / "+string(T.mode);
labels=labels+" / N="+string(T.n_loaded);
end

function save_plot(fig,folder,name,editable)
exportgraphics(fig,fullfile(folder,name+".png"),'Resolution',300);
if editable, savefig(fig,fullfile(folder,name+".fig")); end
end
