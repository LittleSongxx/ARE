let pptxgen;
try {
  pptxgen = require('pptxgenjs');
} catch (error) {
  pptxgen = require('/tmp/pptxgen/node_modules/pptxgenjs');
}

const fs = require('fs');
const path = require('path');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = '宋恩圣';
pptx.subject = '面向大规模自主探索的不确定性感知预测信念图强化学习';
pptx.title = 'UPBG-RL：论文改进计划';
pptx.company = 'Research';
pptx.lang = 'zh-CN';
pptx.theme = {
  headFontFace: '微软雅黑',
  bodyFontFace: '微软雅黑',
  lang: 'zh-CN',
};

pptx.defineSlideMaster({
  title: 'MASTER',
  background: { color: 'FFFFFF' },
  objects: [
    { rect: { x: 0, y: 0, w: 13.333, h: 0.08, fill: { color: '0F6FC6' }, line: { color: '0F6FC6', transparency: 100 } } },
    { line: { x: 0.55, y: 7.05, w: 12.25, h: 0, line: { color: 'C7D3DC', width: 0.6 } } },
    { text: { text: 'UPBG-RL  |  论文改进计划', options: { x: 0.58, y: 7.10, w: 5.8, h: 0.18, fontFace: 'Arial', fontSize: 7.5, color: '6B7882', margin: 0, breakLine: false } } },
  ],
  slideNumber: { x: 12.42, y: 7.08, color: '0F6FC6', fontFace: 'Arial', fontSize: 8 },
});

const C = {
  bg: 'FFFFFF', panel: 'F7F9FB', panel2: 'EAF4F8', line: 'C7D3DC',
  text: '1A1C1E', muted: '52616B', dim: '7B8790', blue: '0F6FC6',
  blue2: '7DA5CB', cyan: '009DD9', green: '10CF9B', yellow: 'A5C249',
  orange: 'F49100', red: 'E13B3B', purple: '628EE3', teal: '04617B', black: '1A1C1E',
};
const F = '微软雅黑';
const A = 'Arial';
const asset = (name) => path.join(__dirname, 'assets', name);
const mapAsset = (name) => path.join(__dirname, '..', 'maps', name);

function addText(slide, text, x, y, w, h, opts = {}) {
  slide.addText(text, {
    x, y, w, h,
    fontFace: opts.fontFace || F,
    fontSize: opts.fontSize || 14,
    color: opts.color || C.text,
    margin: opts.margin === undefined ? 0 : opts.margin,
    breakLine: opts.breakLine === undefined ? false : opts.breakLine,
    fit: 'shrink',
    valign: opts.valign || 'mid',
    paraSpaceAfterPt: 0,
    charSpacing: opts.charSpacing || 0,
    bold: opts.bold || false,
    italic: opts.italic || false,
    align: opts.align || 'left',
    bullet: opts.bullet,
    transparency: opts.transparency || 0,
  });
}

function rect(slide, x, y, w, h, fill, radius = 0.08, lineColor = C.line, transparency = 0) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x, y, w, h,
    rectRadius: radius,
    fill: { color: fill, transparency },
    line: { color: lineColor === 'none' ? fill : lineColor, width: 0.8, transparency: lineColor === 'none' ? 100 : 0 },
  });
}

function line(slide, x1, y1, x2, y2, color = C.line, width = 1, dash = 'solid', begin = '', end = '') {
  slide.addShape(pptx.ShapeType.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color, width, dashType: dash, beginArrowType: begin, endArrowType: end },
  });
}

function circle(slide, x, y, d, fill, lineColor = fill, transparency = 0) {
  slide.addShape(pptx.ShapeType.ellipse, {
    x, y, w: d, h: d,
    fill: { color: fill, transparency },
    line: { color: lineColor === 'none' ? fill : lineColor, width: 0.8, transparency: lineColor === 'none' ? 100 : 0 },
  });
}

function pill(slide, text, x, y, w, color = C.blue, fill = C.panel2) {
  rect(slide, x, y, w, 0.28, fill, 0.14, fill);
  addText(slide, text, x + 0.04, y + 0.01, w - 0.08, 0.23, { fontSize: 8.5, color, bold: true, align: 'center' });
}

function title(slide, kicker, heading, sub = '') {
  rect(slide, 0.46, 0.64, 0.055, 0.50, C.blue, 0.01, C.blue);
  rect(slide, 0.46, 1.14, 0.055, 0.10, C.green, 0.01, C.green);
  addText(slide, kicker.toUpperCase(), 0.62, 0.34, 4.2, 0.22, { fontFace: A, fontSize: 9, color: C.blue, bold: true, charSpacing: 1.3 });
  addText(slide, heading, 0.60, 0.64, 12.0, 0.52, { fontSize: 23, color: C.text, bold: true });
  if (sub) addText(slide, sub, 0.62, 1.22, 11.8, 0.30, { fontSize: 10.5, color: C.muted });
  line(slide, 0.62, 1.62, 12.72, 1.62, C.line, 0.8);
}

function source(slide, text) {
  addText(slide, text, 6.80, 6.83, 5.80, 0.17, { fontFace: A, fontSize: 6.8, color: C.dim, align: 'right' });
}

function note(slide, text) {
  if (typeof slide.addNotes === 'function') slide.addNotes(text);
}

function img(slide, file, x, y, w, h, transparency = 0) {
  if (fs.existsSync(file)) slide.addImage({ path: file, x, y, w, h, transparency });
}

function arrow(slide, x1, y1, x2, y2, color = C.blue, width = 1.4, dash = 'solid') {
  if (Math.abs(y2 - y1) < 0.001 && x2 > x1) {
    const gap = x2 - x1;
    const d = Math.min(0.18, Math.max(0.12, gap * 0.55));
    slide.addShape(pptx.ShapeType.chevron, {
      x: (x1 + x2 - d) / 2,
      y: y1 - d / 2,
      w: d,
      h: d,
      fill: { color },
      line: { color, transparency: 100 },
    });
    return;
  }
  line(slide, x1, y1, x2, y2, color, width, dash, '', 'triangle');
}

function node(slide, x, y, label, fill = C.panel2, stroke = C.blue, w = 1.15, h = 0.58, fs = 10, color = C.text) {
  rect(slide, x, y, w, h, fill, 0.08, stroke);
  addText(slide, label, x + 0.05, y + 0.07, w - 0.10, h - 0.11, { fontSize: fs, color, bold: true, align: 'center', breakLine: true });
}

function card(slide, x, y, w, h, heading, body, accent = C.blue, opts = {}) {
  rect(slide, x, y, w, h, opts.fill || C.panel, 0.08, opts.line || C.line);
  rect(slide, x, y, 0.055, h, accent, 0.02, accent);
  addText(slide, heading, x + 0.22, y + 0.15, w - 0.38, 0.30, { fontSize: opts.headingSize || 12.3, color: opts.headingColor || C.text, bold: true });
  if (body) addText(slide, body, x + 0.22, y + 0.54, w - 0.40, h - 0.64, { fontSize: opts.bodySize || 9.5, color: opts.bodyColor || C.muted, valign: 'top', breakLine: true });
}

function stat(slide, value, label, x, y, w, color = C.blue) {
  addText(slide, value, x, y, w, 0.44, { fontFace: A, fontSize: 24, color, bold: true });
  addText(slide, label, x, y + 0.46, w, 0.36, { fontSize: 9.3, color: C.muted, valign: 'top', breakLine: true });
}

function tableHeader(slide, headers, x0, y, widths) {
  let x = x0;
  headers.forEach((header, index) => {
    rect(slide, x, y, widths[index], 0.48, C.teal, 0.03, C.teal);
    addText(slide, header, x + 0.12, y + 0.13, widths[index] - 0.24, 0.18, { fontSize: 9.8, color: 'FFFFFF', bold: true });
    x += widths[index];
  });
}

// 1. Cover
{
  const s = pptx.addSlide('MASTER');
  s.background = { color: C.bg };
  rect(s, 0, 2.32, 13.333, 2.30, C.blue, 0, C.blue);
  rect(s, 0, 2.32, 4.20, 2.30, C.teal, 0, C.teal, 5);
  addText(s, '论文改进计划 / 2026.08', 0.72, 0.80, 3.8, 0.28, { fontSize: 12, color: C.blue, bold: true, charSpacing: 1.0 });
  addText(s, 'UPBG-RL', 0.72, 2.56, 11.90, 0.54, { fontFace: A, fontSize: 31, color: 'FFFFFF', bold: true, align: 'center' });
  addText(s, 'Uncertainty-aware Predictive Belief Graph Reinforcement Learning', 0.72, 3.16, 11.90, 0.31, { fontFace: A, fontSize: 14, color: 'FFFFFF', bold: true, align: 'center' });
  addText(s, '面向大规模自主探索的预测式多尺度信念图强化学习', 0.72, 3.66, 11.90, 0.35, { fontSize: 17, color: 'FFFFFF', bold: true, align: 'center' });
  addText(s, '基于 ARiADNE / Large-scale DRL baseline 的论文主线升级', 0.72, 4.12, 11.90, 0.22, { fontSize: 10.5, color: 'DBF5F9', align: 'center' });
  img(s, mapAsset('img_412.png'), 10.08, 4.83, 2.25, 1.69, 28);
  line(s, 0.72, 5.05, 9.30, 5.05, C.blue2, 1.0);
  addText(s, '汇报人：宋恩圣', 0.72, 5.34, 2.6, 0.28, { fontSize: 12, color: C.text, bold: true });
  addText(s, '导师：梁建', 0.72, 5.74, 2.6, 0.25, { fontSize: 10.5, color: C.muted });
  addText(s, '核心转变：不只看眼前 frontier，而是判断哪条路更可能打开新区域', 3.20, 5.38, 6.05, 0.46, { fontSize: 11, color: C.blue, align: 'right', breakLine: true });
  addText(s, '论文改进计划汇报', 0.72, 6.42, 3.2, 0.22, { fontSize: 9.2, color: C.dim });
  note(s, '开场先讲清楚核心变化：baseline 擅长根据当前图选择下一步，本计划进一步让策略显式判断候选方向后面可能打开多少新空间，并知道这份判断是否可靠。');
}

// 2. Problem
{
  const s = pptx.addSlide('MASTER');
  title(s, '01 / Research question', '探索真正困难的部分：评价“尚未看见”的区域', '机器人只能看到局部地图，却要选择对最终路径最有利的下一步；当前收益与长期潜力并不等价。');
  rect(s, 0.68, 2.04, 7.18, 4.10, C.panel, 0.08, C.line);
  addText(s, '部分可观测的闭环决策', 0.96, 2.28, 3.0, 0.28, { fontSize: 13.5, color: C.text, bold: true });
  const loopX = [1.00, 2.72, 4.44, 6.16];
  const loopItems = [
    ['局部地图', '当前已知范围', C.blue],
    ['候选节点', '眼前可见信息', C.green],
    ['选择下一步', '输出导航目标', C.orange],
    ['执行并更新', '获得新的观测', C.yellow],
  ];
  loopItems.forEach((item, i) => {
    node(s, loopX[i], 3.18, item[0], C.panel2, item[2], 1.34, 0.66, 9.8);
    addText(s, item[1], loopX[i] - 0.02, 4.02, 1.38, 0.24, { fontSize: 8.7, color: C.muted, align: 'center' });
    if (i < loopItems.length - 1) arrow(s, loopX[i] + 1.42, 3.51, loopX[i + 1] - 0.09, 3.51, C.blue2, 1.1);
  });
  addText(s, '地图更新后进入下一轮判断', 1.02, 4.55, 6.47, 0.24, { fontSize: 9.2, color: C.dim, align: 'center' });
  rect(s, 1.02, 5.03, 6.46, 0.66, 'FFF3D8', 0.06, C.orange);
  addText(s, '难点：眼前 frontier 多，不代表这条路后面真的有更多可探索空间。', 1.28, 5.21, 5.94, 0.28, { fontSize: 10.2, color: C.orange, bold: true, align: 'center' });
  addText(s, '目标：完成覆盖，同时减少总路程、回退和重复访问', 1.08, 5.88, 6.25, 0.22, { fontSize: 10, color: C.blue, bold: true, align: 'center' });
  addText(s, '本文核心问题', 8.24, 2.22, 3.4, 0.28, { fontSize: 13.5, color: C.text, bold: true });
  card(s, 8.18, 2.72, 4.40, 1.02, '未来收益', '能否显式预测每个节点将揭示多少未知自由空间，而不只依赖当前 frontier 数量？', C.blue, { bodySize: 9.1 });
  card(s, 8.18, 3.80, 4.40, 1.02, '预测可信度', '能否让策略知道“预测有多可靠”，并在保守与主动探索之间自适应切换？', C.green, { bodySize: 9.1 });
  card(s, 8.18, 4.88, 4.40, 1.02, '长程结构', '能否在固定节点预算下，同时保留区域级趋势和 frontier / 路口级局部细节？', C.orange, { bodySize: 9.1 });
  source(s, 'Problem formulation: Cao et al. (2023, 2024)');
  note(s, '先从任务本质切入：探索是对未知区域价值的序列估计问题。后续三项方法分别回答未来收益、可信度和长程结构。');
}

// 3. Baseline lineage
{
  const s = pptx.addSlide('MASTER');
  title(s, '02 / Baseline', '两代 baseline 已经解决了什么，我们还缺什么', 'ARiADNE 建立图注意力决策主干；2024 年工作用训练期真值 Critic 与图稀疏化扩展到大场景。');
  line(s, 1.08, 3.25, 12.10, 3.25, C.line, 2);
  const marks = [[1.42, C.blue, '2023'], [6.03, C.green, '2024'], [10.72, C.orange, '本计划']];
  marks.forEach((m) => { circle(s, m[0], 3.04, 0.42, m[1], m[1]); addText(s, m[2], m[0] - 0.30, 2.43, 1.10, 0.28, { fontFace: A, fontSize: 16, color: m[1], bold: true }); });
  card(s, 0.82, 3.64, 3.62, 1.78, 'ARiADNE / ICRA 2023', '把可通行区域转换为图节点\n用多层注意力理解局部与全局关系\n离散 SAC 选择下一导航点', C.blue, { bodySize: 9.5 });
  card(s, 4.76, 3.64, 3.62, 1.78, 'Large-scale DRL / RA-L 2024', '训练时让 Critic 看完整地图，减少估值噪声\n图稀疏化控制大场景计算量\n完成大场景仿真和真实机器人验证', C.green, { bodySize: 9.5 });
  card(s, 8.70, 3.64, 3.82, 1.78, 'UPBG-RL / 本文计划', '显式学习候选方向后面的未来空间潜力\n跨时间维护判断及其可信度\n用区域级和局部级图同时支持远近决策', C.orange, { bodySize: 9.5 });
  pill(s, '主干保留', 1.10, 5.82, 1.10, C.blue, C.panel2);
  addText(s, '动作定义、在线重规划、注意力 Actor 与 SAC 训练框架', 2.38, 5.82, 4.80, 0.28, { fontSize: 10.2, color: C.muted });
  pill(s, '研究升级', 8.08, 5.82, 1.10, C.orange, 'FFF3D8');
  addText(s, '从隐式估值升级为可解释的未来潜力判断', 9.36, 5.82, 2.90, 0.28, { fontSize: 10.2, color: C.muted });
  source(s, 'Cao et al., ICRA 2023; IEEE RA-L 2024');
  note(s, '强调继承关系。我们的贡献不是重做 actor-critic，而是改变 actor 用来推理未知区域的状态表示和监督方式。');
}

// 4. Baseline 1: graph decision process
{
  const s = pptx.addSlide('MASTER');
  title(s, '03 / Baseline 1', 'ARiADNE：把探索改写成图上的连续选点', '它不一次规划完整轨迹，而是反复选择一个可行邻居作为下一视点，边走边更新地图。');
  rect(s, 0.72, 2.02, 5.18, 4.08, C.panel, 0.08, C.line);
  img(s, asset('ariadne_decision_graph.png'), 1.08, 2.28, 4.46, 3.54);
  addText(s, '灰色：未知区域　白色：已知自由区域　彩色点：候选视点与当前 utility', 1.00, 5.84, 4.62, 0.20, { fontSize: 8.3, color: C.dim, align: 'center' });
  addText(s, '一次决策如何形成', 6.26, 2.06, 2.40, 0.30, { fontSize: 14, color: C.text, bold: true });
  const steps = [
    ['01', '候选视点', '在已知自由区域均匀布点，形成机器人可能到达的位置。', C.blue],
    ['02', '无碰撞连边', '只连接近邻且路径不穿过障碍或未知区域，得到可执行图。', C.green],
    ['03', '节点信息', '每个点包含位置、当前可见 frontier 数量和是否访问。', C.orange],
    ['04', '下一动作', '策略从当前点的可行邻居中选一个目标；到达后重新建图。', C.yellow],
  ];
  steps.forEach((item, i) => {
    const y = 2.50 + i * 0.80;
    rect(s, 6.24, y, 6.18, 0.64, i % 2 === 0 ? C.panel2 : C.panel, 0.06, C.line);
    circle(s, 6.48, y + 0.17, 0.30, item[3], 'none');
    addText(s, item[0], 6.48, y + 0.26, 0.30, 0.10, { fontFace: A, fontSize: 7.2, color: 'FFFFFF', bold: true, align: 'center' });
    addText(s, item[1], 6.94, y + 0.16, 1.30, 0.24, { fontSize: 10.5, color: item[3], bold: true });
    addText(s, item[2], 8.34, y + 0.11, 3.72, 0.42, { fontSize: 9.2, color: C.muted, breakLine: true });
  });
  rect(s, 6.24, 5.82, 6.18, 0.40, 'E3F4EB', 0.05, C.green);
  addText(s, '闭环：局部地图 → 动态图 → 下一视点 → 新观测 → 更新地图', 6.52, 5.92, 5.62, 0.20, { fontSize: 9.6, color: C.teal, bold: true, align: 'center' });
  source(s, 'Reused figure from the original monthly report; Cao et al., ICRA 2023, Fig. 2');
  note(s, '先把 ARiADNE 讲成一个闭环：地图只决定当前可行图，策略每次只选一个邻居。utility 是该点当前能看到的 frontier 数量，访问标记让策略知道自己走过哪里。');
}

// 5. Baseline 1: policy and training
{
  const s = pptx.addSlide('MASTER');
  title(s, '04 / Baseline 1', 'ARiADNE：注意力网络怎样做出“非短视”选择', 'Encoder 理解整张局部图的区域依赖，Decoder 再把全局判断落到当前可行邻居。');
  rect(s, 0.72, 2.02, 7.96, 2.84, C.panel, 0.08, C.line);
  img(s, asset('ariadne_policy_network.png'), 0.98, 2.40, 7.44, 2.20);
  card(s, 8.92, 2.02, 3.66, 1.32, 'Encoder / 看懂整张图', '6 层图注意力逐层聚合邻居；堆叠后，每个节点都带上更远区域的上下文。', C.blue, { bodySize: 9.2 });
  card(s, 8.92, 3.54, 3.66, 1.32, 'Decoder / 落到下一步', '当前位置先读取全图，再只比较可行邻居；Pointer 输出各邻居的选择概率。', C.green, { bodySize: 9.2 });
  const train = [
    ['Actor / 负责选择', '根据局部信息图输出下一导航点；部署时保留。', C.blue],
    ['Critic / 负责评估', '预测动作的长期累计回报，用来训练 Actor；部署时移除。', C.green],
    ['SAC / 负责学习', '发现新 frontier 得分，移动距离扣分，完成探索再加分。', C.orange],
  ];
  train.forEach((item, i) => card(s, 0.72 + i * 4.02, 5.14, 3.72, 1.04, item[0], item[1], item[2], { bodySize: 9.1, headingSize: 11.6 }));
  rect(s, 0.72, 6.32, 11.76, 0.34, 'FFF3D8', 0.05, C.orange);
  addText(s, '非短视性来自“全图注意力 + 长期回报估计”；但未知区域的未来价值仍然只是隐式学到。', 0.98, 6.41, 11.24, 0.17, { fontSize: 9.4, color: C.orange, bold: true, align: 'center' });
  source(s, 'Reused architecture from the original monthly report; Cao et al., ICRA 2023, Fig. 3');
  note(s, 'Encoder 负责把邻域信息传播到更远位置，Decoder 先理解全图再在可行邻居中做选择。Critic 通过长期累计奖励告诉 Actor 哪个动作从长远看更好，但这个长期潜力没有单独的监督目标。');
}

// 6. Baseline 2: privileged critic
{
  const s = pptx.addSlide('MASTER');
  title(s, '05 / Baseline 2', 'Large-scale DRL：先解决训练噪声，再谈大场景', '第二篇保留 ARiADNE 的图注意力 Actor，新增训练期真值 Critic 与图稀疏化。');
  card(s, 0.72, 2.02, 5.70, 1.26, '问题 A：Critic 只能看见局部地图', '探索是部分可观测问题。Critic 一边猜未知区域，一边估计长期回报，训练目标容易波动，进而影响 Actor。', C.blue, { bodySize: 9.5 });
  card(s, 6.68, 2.02, 5.70, 1.26, '问题 B：大地图让图迅速膨胀', '节点数量随自由空间增加；远处 frontier 之间夹着大量零 utility 节点，计算变重，信息传播也变长。', C.orange, { bodySize: 9.5 });
  rect(s, 0.72, 3.62, 5.70, 2.26, C.panel2, 0.08, C.blue2);
  addText(s, 'Actor：始终只看局部信息', 0.98, 3.88, 2.74, 0.28, { fontSize: 13, color: C.blue, bold: true });
  const actorX = [1.00, 2.86, 4.72];
  const actorSteps = ['局部信息图', '提出下一动作', '更新选择策略'];
  actorSteps.forEach((label, i) => {
    node(s, actorX[i], 4.54, label, 'FFFFFF', [C.blue, C.green, C.orange][i], 1.34, 0.60, 9.2);
    if (i < 2) arrow(s, actorX[i] + 1.43, 4.84, actorX[i + 1] - 0.09, 4.84, C.blue2, 1.1);
  });
  addText(s, '部署条件没有变化', 1.06, 5.40, 4.92, 0.22, { fontSize: 9.2, color: C.blue, bold: true, align: 'center' });
  rect(s, 6.68, 3.62, 5.70, 2.26, 'F1EDF9', 0.08, C.purple);
  addText(s, 'Critic：训练时读取完整地图', 6.94, 3.88, 3.10, 0.28, { fontSize: 13, color: C.purple, bold: true });
  const criticX = [6.96, 8.82, 10.68];
  const criticSteps = ['完整地图图', '估计长期价值', '提供稳定目标'];
  criticSteps.forEach((label, i) => {
    node(s, criticX[i], 4.54, label, 'FFFFFF', C.purple, 1.34, 0.60, 9.2);
    if (i < 2) arrow(s, criticX[i] + 1.43, 4.84, criticX[i + 1] - 0.09, 4.84, C.purple, 1.1);
  });
  addText(s, '论文报告价值损失与梯度方差约降低一个数量级', 7.02, 5.40, 4.92, 0.22, { fontSize: 9.2, color: C.purple, bold: true, align: 'center' });
  rect(s, 0.72, 6.12, 11.66, 0.40, 'E3F4EB', 0.05, C.green);
  addText(s, '完整地图只帮助 Critic 在训练中教得更稳；部署时 Critic 与真值图全部移除。', 0.98, 6.22, 11.14, 0.20, { fontSize: 9.7, color: C.teal, bold: true, align: 'center' });
  source(s, 'Cao et al., IEEE RA-L 2024, Sec. IV-C and Appendix Fig. 8');
  note(s, 'Baseline2 的第一项变化不是让机器人运行时偷看完整地图，而是利用 Actor-Critic 的训练结构：Critic 本来就只在训练时存在，因此可用真值图给出更稳定的长期价值目标。');
}

// 7. Baseline 2: graph rarefaction
{
  const s = pptx.addSlide('MASTER');
  title(s, '06 / Baseline 2', '图稀疏化：把小场景训练出的策略带到大场景', '目标不是随意删点，而是用更少节点保留 frontier、关键通路和远程连通关系。');
  rect(s, 0.72, 2.02, 7.34, 3.54, C.panel, 0.08, C.line);
  img(s, asset('large_p3_fig.png'), 0.90, 2.24, 6.98, 2.80);
  addText(s, '点云 → Octomap → 信息图 → 稀疏图 → 注意力策略', 1.10, 5.20, 6.58, 0.20, { fontSize: 9.0, color: C.blue, bold: true, align: 'center' });
  const rare = [
    ['01', '聚合探索目标', '把非零 utility 节点按邻接关系组成若干信息区域。', C.blue],
    ['02', '寻找通往目标的骨架', '从机器人到每个区域运行 A*，得到需要保留的通路。', C.green],
    ['03', '视线压缩路径节点', '能直达时跳过中间点，只留下转折与连接所需节点。', C.orange],
    ['04', '重建稀疏信息图', '用更短的图距离连接远端信息，同一 Actor 无需重新训练。', C.yellow],
  ];
  rare.forEach((item, i) => {
    const y = 2.05 + i * 0.86;
    rect(s, 8.30, y, 4.20, 0.70, i % 2 === 0 ? C.panel2 : C.panel, 0.06, C.line);
    pill(s, item[0], 8.50, y + 0.20, 0.56, item[3], i === 2 ? 'FFF3D8' : 'FFFFFF');
    addText(s, item[1], 9.24, y + 0.13, 1.42, 0.24, { fontSize: 10.2, color: item[3], bold: true });
    addText(s, item[2], 10.68, y + 0.08, 1.62, 0.54, { fontSize: 8.4, color: C.muted, breakLine: true });
  });
  rect(s, 0.72, 5.86, 11.78, 0.46, C.panel2, 0.06, C.blue2);
  addText(s, 'Small-to-large 的关键：训练时学会“如何在图上比较方向”，部署时再用稀疏图控制规模和传播距离。', 0.98, 5.98, 11.26, 0.22, { fontSize: 9.8, color: C.blue, bold: true, align: 'center' });
  source(s, 'Cao et al., IEEE RA-L 2024, Fig. 2 and Sec. IV-D');
  note(s, '图稀疏化先找到所有有信息的区域，再保留从机器人到这些区域的关键骨架，最后重建稀疏连接。这样不是改变策略，而是把大场景转换为策略熟悉的较短图结构。');
}

// 8. Baseline evidence
{
  const s = pptx.addSlide('MASTER');
  title(s, '07 / Baseline evidence', '两次 baseline 升级分别带来了什么', 'ARiADNE 先证明图注意力策略能减少短视；baseline2 再通过真值 Critic 与稀疏图扩展训练稳定性和场景规模。');
  img(s, asset('large_p6_fig.png'), 0.70, 2.02, 5.12, 3.75);
  rect(s, 0.70, 5.89, 5.12, 0.36, C.panel2, 0.05, C.panel2);
  addText(s, '130 m × 100 m indoor benchmark', 0.92, 5.96, 4.70, 0.18, { fontFace: A, fontSize: 8.8, color: C.muted, align: 'center' });
  addText(s, '论文报告的关键结果', 6.36, 2.08, 3.0, 0.30, { fontSize: 14, color: C.text, bold: true });
  stat(s, '5%', 'ARiADNE random set：平均路径长度优于 TARE Local', 6.38, 2.60, 2.55, C.blue);
  stat(s, '7%', 'small-scale：真值 Critic 相比 ARiADNE 进一步缩短路径', 9.30, 2.60, 2.55, C.green);
  stat(s, '12%', 'large-scale indoor：distance efficiency 优于 TARE', 6.38, 3.82, 2.55, C.yellow);
  stat(s, '60% faster', '论文对 large-scale planning computation 的表述', 9.30, 3.82, 2.55, C.orange);
  card(s, 6.36, 5.08, 6.03, 1.12, '能力已经很强，但未知区域仍靠隐式猜测', '室内大场景中路径与计算效率更优，也完成真实机器人验证；但室外结构变化时仍会随机选点，说明未来潜力缺少显式表达与可信度。', C.red, { bodySize: 9.2 });
  source(s, 'Cao et al., ICRA 2023, Table I; IEEE RA-L 2024, Tables I–III and Sec. V');
  note(s, '用四个数字串起两篇论文：ARiADNE 相比 TARE Local 约提升 5%；baseline2 的真值 Critic 又比 ARiADNE 缩短约 7%；进入室内大场景后，距离效率仍优于 TARE，并保持更快规划。最后落到本文缺口：未知区域潜力仍靠隐式预测。');
}

// 9. Gaps
{
  const s = pptx.addSlide('MASTER');
  title(s, '08 / Research gap', '缺的不是更多模块，而是一份“未知区域价值判断”', '现有方法能根据局部图做决策，但对候选方向的长期潜力仍主要依赖隐式学习。');
  const gaps = [
    ['缺口 1 / 只看当前', '未来潜力没有直接教学答案', 'utility 只统计当前能看到的 frontier；未知区域的长期价值主要通过最终奖励间接学到，信号较稀疏。', C.blue],
    ['缺口 2 / 缺少记忆', '跨时间判断容易不一致', '同一节点的判断会随新地图跳变；策略不知道本次预测有多可靠，也难区分观测噪声和真实结构变化。', C.green],
    ['缺口 3 / 远近混合', '全局趋势与局部细节混在一起', '统一的邻域传播与固定稀疏规则，难以同时保留远处区域趋势和近处 frontier 细节。', C.orange],
    ['缺口 4 / 教得不够', '完整地图知识主要停留在 Critic', '完整地图帮助 Critic 稳定估值，却没有直接教 Actor 识别节点的长期潜力和远近结构。', C.purple],
  ];
  gaps.forEach((g, i) => {
    const x = 0.76 + (i % 2) * 6.02;
    const y = 2.06 + Math.floor(i / 2) * 1.74;
    card(s, x, y, 5.72, 1.40, g[1], g[2], g[3], { bodySize: 9.6 });
    pill(s, g[0], x + 3.92, y + 0.16, 1.42, g[3], i === 2 ? 'FFF3D8' : C.panel2);
  });
  rect(s, 0.76, 5.72, 11.80, 0.52, C.panel2, 0.06, C.blue2);
  addText(s, '研究命题：让策略显式判断“从这里继续走，可能打开多少新空间”，并持续维护判断的可信度。', 1.02, 5.86, 11.28, 0.22, { fontSize: 11, color: C.blue, bold: true, align: 'center' });
  source(s, 'Research gap derived from baseline inputs and stated future work');
  note(s, '四个缺口实际指向同一个问题：策略缺少一份可解释、可更新的未来收益判断。后续方法都围绕这份判断展开。');
}

// 10. Final idea
{
  const s = pptx.addSlide('MASTER');
  title(s, '09 / Final idea', '最终优化思路：给每个候选方向一份“未来潜力判断”', '这份判断既回答“后面可能有多少新空间”，也回答“我对这个判断有多大把握”。');
  rect(s, 0.76, 2.02, 11.82, 0.72, C.panel2, 0.06, C.blue2);
  addText(s, '机器人不再只比较眼前有多少 frontier，而是比较：继续走下去，哪条路更可能带来真正有价值的新区域。', 1.08, 2.22, 11.18, 0.30, { fontSize: 12.3, color: C.blue, bold: true, align: 'center' });
  const ideas = [
    ['1', '先看得更远', '从局部地图和图结构中，预测候选节点后面可能打开多少尚未发现的自由空间。', C.blue],
    ['2', '再判断是否可信', '把历史判断和当前证据放在一起看；证据可靠就跟进，结构突变就重新评估。', C.green],
    ['3', '最后兼顾远近', '远处保留区域方向，近处保留 frontier、路口与可执行路径，让长期目标能落到下一步动作。', C.orange],
  ];
  const ideaX = [0.78, 4.89, 9.00];
  ideas.forEach((item, i) => {
    rect(s, ideaX[i], 3.16, 3.55, 2.14, C.panel, 0.08, item[3]);
    circle(s, ideaX[i] + 0.26, 3.45, 0.46, item[3], 'none');
    addText(s, item[0], ideaX[i] + 0.26, 3.59, 0.46, 0.12, { fontFace: A, fontSize: 9, color: 'FFFFFF', bold: true, align: 'center' });
    addText(s, item[1], ideaX[i] + 0.88, 3.45, 2.22, 0.30, { fontSize: 13, color: item[3], bold: true });
    addText(s, item[2], ideaX[i] + 0.28, 4.05, 2.98, 0.84, { fontSize: 10, color: C.muted, breakLine: true, valign: 'top' });
    if (i < ideas.length - 1) arrow(s, ideaX[i] + 3.64, 4.22, ideaX[i + 1] - 0.10, 4.22, C.blue2, 1.2);
  });
  rect(s, 0.78, 5.74, 11.77, 0.54, 'E3F4EB', 0.06, C.green);
  addText(s, '最终进入策略的不是三套零散模块，而是一份会预测、会更新、能跨尺度表达的未知区域价值判断。', 1.04, 5.89, 11.25, 0.24, { fontSize: 10.6, color: C.teal, bold: true, align: 'center' });
  source(s, 'UPBG-RL proposed method');
  note(s, '这一页只讲最终方法，不讲思想来自哪里。用一句话概括：先判断候选方向的长期潜力，再根据可信度和地图尺度把判断转成下一步动作。');
}

// 11. Overall architecture
{
  const s = pptx.addSlide('MASTER');
  title(s, '10 / Overall method', '完整方法只有四步：预测、更新、组织、决策', '四步沿同一条部署链运行；机器人实际执行时仍然只读取自己的局部地图。');
  const flowX = [0.72, 3.79, 6.86, 9.93];
  const flow = [
    ['01', '异方差潜力预测', '用特权信息蒸馏学习 future-gain，同时输出潜力均值与预测不确定性。', C.blue],
    ['02', '自适应 KF 更新', '将潜力预测作为观测；由不确定性和结构事件自适应调整 Q/R。', C.green],
    ['03', '图小波 + Pooling', 'Random-Walk Graph Wavelet 分离低频趋势和高频细节，再重组节点。', C.orange],
    ['04', '层级 Pointer 决策', '区域上下文先定方向，Decoder 再从当前可行邻居中选下一航点。', C.yellow],
  ];
  flow.forEach((item, i) => {
    rect(s, flowX[i], 2.18, 2.50, 2.22, C.panel, 0.08, item[3]);
    pill(s, item[0], flowX[i] + 0.22, 2.43, 0.62, item[3], i === 2 ? 'FFF3D8' : C.panel2);
    addText(s, item[1], flowX[i] + 0.22, 2.91, 2.02, 0.30, { fontSize: 13, color: item[3], bold: true });
    addText(s, item[2], flowX[i] + 0.22, 3.35, 2.04, 0.74, { fontSize: 9.6, color: C.muted, breakLine: true, valign: 'top' });
    if (i < flow.length - 1) arrow(s, flowX[i] + 2.59, 3.29, flowX[i + 1] - 0.10, 3.29, C.blue2, 1.2);
  });
  rect(s, 0.72, 4.86, 11.71, 0.64, C.panel2, 0.06, C.blue2);
  addText(s, '部署阶段', 0.98, 5.04, 1.26, 0.24, { fontSize: 10.5, color: C.blue, bold: true });
  addText(s, '局部地图进入策略，直接输出下一导航点；传感器条件与 baseline 一致。', 2.36, 5.03, 9.66, 0.26, { fontSize: 10.2, color: C.muted });
  rect(s, 0.72, 5.72, 11.71, 0.64, 'F1EDF9', 0.06, C.purple);
  addText(s, '训练阶段', 0.98, 5.90, 1.26, 0.24, { fontSize: 10.5, color: C.purple, bold: true });
  addText(s, '完整地图只用来生成“正确答案”和教师提示；训练结束后全部移除。', 2.36, 5.89, 9.66, 0.26, { fontSize: 10.2, color: C.muted });
  source(s, 'Proposed deployment and training boundaries');
  note(s, '先顺着四个水平步骤讲算法：异方差 Potential Head 预测潜力和不确定性；自适应 Kalman Filter 维护时序 belief；Random-Walk Graph Wavelet 与 Pooling 重组远近尺度；层级 Pointer 最终只在可行邻居中选一步。完整地图只在训练期提供蒸馏教师和标签，上线后没有额外传感器。');
}

// 12. Algorithm blueprint
{
  const s = pptx.addSlide('MASTER');
  title(s, '11 / Algorithm blueprint', '算法蓝图：蒸馏、Kalman 与 Graph Wavelet 怎样进入决策链', '保留 baseline 的图注意力 Actor 与离散 SAC；三项改进均对应明确的经典算法接口。');
  const methods = [
    {
      x: 0.72, color: C.blue, tag: '算法 A', heading: '特权蒸馏 + 异方差回归',
      steps: [
        ['1', 'GT rollout 构造 future-gain 标签'],
        ['2', 'Teacher–Student 特权信息蒸馏'],
        ['3', 'Gaussian NLL + RankNet 排序监督'],
      ],
      output: 'Potential Head：均值 + 预测方差',
    },
    {
      x: 4.55, color: C.green, tag: '算法 B', heading: '自适应 Kalman Filter',
      steps: [
        ['1', 'Potential Head 输出作为 KF 观测'],
        ['2', '预测方差设 R，结构事件调 Q'],
        ['3', 'Kalman 更新；突变时重置 belief'],
      ],
      output: '状态：未来潜力 + 协方差',
    },
    {
      x: 8.38, color: C.orange, tag: '算法 C', heading: 'Wavelet + DiffPool',
      steps: [
        ['1', '随机游走小波分解：scale 1/2/4'],
        ['2', '低频进 DiffPool；高频保留局部细节'],
        ['3', 'Top-K 选点并保留 A* 路径骨架'],
      ],
      output: '输出：区域 token + 动作节点',
    },
  ];
  methods.forEach((method) => {
    rect(s, method.x, 2.02, 3.55, 3.24, C.panel, 0.08, method.color);
    pill(s, method.tag, method.x + 0.24, 2.27, 0.76, method.color, method.color === C.orange ? 'FFF3D8' : C.panel2);
    addText(s, method.heading, method.x + 1.14, 2.28, 2.06, 0.28, { fontSize: 12.8, color: method.color, bold: true });
    method.steps.forEach((step, i) => {
      const y = 3.01 + i * 0.58;
      circle(s, method.x + 0.28, y, 0.25, method.color, 'none');
      addText(s, step[0], method.x + 0.28, y + 0.075, 0.25, 0.10, { fontFace: A, fontSize: 7.2, color: 'FFFFFF', bold: true, align: 'center' });
      addText(s, step[1], method.x + 0.66, y - 0.01, 2.52, 0.28, { fontSize: 9.3, color: C.muted, bold: i === 0, breakLine: true });
      if (i < method.steps.length - 1) line(s, method.x + 0.405, y + 0.28, method.x + 0.405, y + 0.53, C.line, 0.8);
    });
    rect(s, method.x + 0.24, 4.82, 3.07, 0.28, method.color === C.orange ? 'FFF3D8' : C.panel2, 0.10, method.color);
    addText(s, method.output, method.x + 0.34, 4.88, 2.87, 0.14, { fontSize: 8.5, color: method.color, bold: true, align: 'center' });
  });
  rect(s, 0.72, 5.66, 11.66, 0.72, C.panel2, 0.06, C.blue2);
  addText(s, '进入 Actor', 0.96, 5.88, 0.86, 0.22, { fontSize: 10.2, color: C.blue, bold: true });
  node(s, 1.96, 5.78, 'utility + KF belief\n+ LF / HF + 几何', 'FFFFFF', C.blue, 2.13, 0.48, 8.6);
  arrow(s, 4.18, 6.02, 4.61, 6.02, C.blue2, 1.0);
  node(s, 4.72, 5.78, '共享 Graph Encoder', 'FFFFFF', C.green, 1.94, 0.48, 8.8);
  arrow(s, 6.75, 6.02, 7.18, 6.02, C.blue2, 1.0);
  node(s, 7.29, 5.78, '区域到局部的注意力', 'FFFFFF', C.orange, 2.02, 0.48, 8.8);
  arrow(s, 9.40, 6.02, 9.83, 6.02, C.blue2, 1.0);
  node(s, 9.94, 5.78, 'Pointer 选择可行邻居', 'FFFFFF', C.yellow, 2.06, 0.48, 8.8);
  source(s, 'Qualitative algorithm blueprint; implementation choices remain to be validated');
  note(s, '算法 A 采用 Teacher–Student 特权信息蒸馏：GT rollout 构造 future-gain，学生 Potential Head 用异方差 Gaussian NLL 学均值与方差，并用 RankNet 式成对排序学习候选顺序。算法 B 采用自适应 Kalman Filter：Potential Head 是观测，预测方差对应观测噪声 R，地图结构事件提高过程噪声 Q 或触发重置。算法 C 采用 Random-Walk Graph Wavelet，在 1/2/4-hop 尺度得到低频趋势与高频残差；低频经 DiffPool 形成区域 token，高频节点以 Top-K 保留，A* 路径骨架保证连通。最后仍由 Encoder、层级 Decoder 和 Pointer 输出可行邻居。');
}

// 13. Future potential
{
  const s = pptx.addSlide('MASTER');
  title(s, '12 / Improvement 1', '改进一：特权信息蒸馏 + 异方差图回归', 'GT rollout 构造 future-gain；Potential Head 用 Gaussian NLL 学潜力与预测方差，用 RankNet 学候选排序。');
  rect(s, 0.72, 2.04, 3.38, 4.12, C.panel, 0.08, C.line);
  addText(s, '算法 A / Privileged Distillation', 1.02, 2.31, 2.62, 0.28, { fontSize: 12.2, color: C.blue, bold: true });
  const labelSteps = [
    ['1', '冻结当前局部地图与候选图'],
    ['2', 'GT 地图上做有限视距 rollout，生成 future-gain 标签'],
    ['3', 'Teacher / Student Graph Encoder 对齐潜力表征'],
    ['4', 'Gaussian NLL 回归 + RankNet 邻居排序'],
  ];
  labelSteps.forEach((step, i) => {
    const y = 2.87 + i * 0.64;
    circle(s, 1.02, y, 0.26, C.blue, 'none');
    addText(s, step[0], 1.02, y + 0.078, 0.26, 0.10, { fontFace: A, fontSize: 7.3, color: 'FFFFFF', bold: true, align: 'center' });
    addText(s, step[1], 1.46, y - 0.02, 2.24, 0.40, { fontSize: 9.1, color: C.muted, bold: i === 3, breakLine: true });
    if (i < labelSteps.length - 1) line(s, 1.15, y + 0.29, 1.15, y + 0.60, C.line, 0.8);
  });
  pill(s, 'Potential Head：均值 + log-variance', 1.02, 5.60, 2.76, C.blue, C.panel2);

  const cases = [
    [4.40, '候选点 A', '眼前 frontier 较多', '短走廊 / 尽头', '当前看起来好，后续空间少', C.orange],
    [8.50, '候选点 B', '眼前 frontier 中等', '相连房间 / 新走廊', '当前不突出，后续空间大', C.blue],
  ];
  cases.forEach((item) => {
    rect(s, item[0], 2.04, 3.78, 3.56, C.panel, 0.08, item[5]);
    addText(s, item[1], item[0] + 0.26, 2.31, 1.50, 0.28, { fontSize: 13.5, color: item[5], bold: true });
    addText(s, item[2], item[0] + 0.26, 2.77, 3.16, 0.24, { fontSize: 9.7, color: C.muted });
    node(s, item[0] + 0.30, 3.35, '当前位置', C.panel2, C.green, 1.10, 0.58, 9.2);
    arrow(s, item[0] + 1.52, 3.64, item[0] + 2.03, 3.64, item[5], 1.2);
    node(s, item[0] + 2.14, 3.35, item[3], item[5] === C.blue ? C.panel2 : 'FFF3D8', item[5], 1.28, 0.58, 8.9);
    addText(s, item[4], item[0] + 0.28, 4.38, 3.20, 0.48, { fontSize: 10.2, color: C.text, bold: true, align: 'center', breakLine: true });
    addText(s, item[5] === C.blue ? '新方法更愿意保留并验证 B' : '只看当前信息容易被 A 吸引', item[0] + 0.30, 5.08, 3.16, 0.24, { fontSize: 9.3, color: item[5], bold: true, align: 'center' });
  });
  rect(s, 4.40, 5.82, 7.88, 0.36, 'E3F4EB', 0.05, C.green);
  addText(s, '联合目标：异方差 Gaussian NLL + RankNet 成对排序 + 特权蒸馏 + 离散 SAC。', 4.66, 5.91, 7.36, 0.18, { fontSize: 9.2, color: C.teal, bold: true, align: 'center' });
  source(s, 'Privileged distillation + heteroscedastic regression + RankNet, proposed');
  note(s, '算法 A 的标签只在训练期计算。固定同一时刻的 belief map，对每个候选方向在 GT map 上做有限视距 rollout，统计新可达自由空间并按路程折扣，形成 future-gain。Teacher 读取 GT graph，Student 只读 belief graph，通过特权信息蒸馏对齐潜力表征。共享 Graph Encoder 后的 Potential Head 输出均值和 log-variance，用异方差 Gaussian NLL 学数值与数据相关不确定性，用 RankNet 式成对损失学习候选顺序，再与离散 SAC 联合优化。部署时只保留 Student。');
}

// 14. Temporal confidence
{
  const s = pptx.addSlide('MASTER');
  title(s, '13 / Improvement 2', '改进二：结构事件门控的自适应卡尔曼滤波', 'KF 状态是“未来潜力 belief”；Potential Head 是观测，预测方差设置 R，地图结构事件自适应调整 Q。');
  const timeX = [0.72, 3.79, 6.86, 9.93];
  const timeFlow = [
    ['状态关联', '按位置、邻接关系和区域归属匹配前后时刻节点', C.blue],
    ['KF 观测', 'Potential Head 的潜力均值作为 z，预测方差作为 R', C.green],
    ['自适应 Q / R', '稳定时 Q 小；结构事件发生时提高 Q 或触发重置', C.orange],
    ['Kalman 更新', '由 Kalman gain 融合先验与当前观测，输出 posterior', C.cyan],
  ];
  timeFlow.forEach((item, i) => {
    rect(s, timeX[i], 2.10, 2.50, 1.56, C.panel, 0.08, item[2]);
    addText(s, item[0], timeX[i] + 0.22, 2.38, 2.02, 0.28, { fontSize: 12.5, color: item[2], bold: true, align: 'center' });
    addText(s, item[1], timeX[i] + 0.22, 2.84, 2.04, 0.54, { fontSize: 9.2, color: C.muted, breakLine: true, align: 'center' });
    if (i < timeFlow.length - 1) arrow(s, timeX[i] + 2.59, 2.88, timeX[i + 1] - 0.10, 2.88, C.blue2, 1.2);
  });
  pill(s, '每个节点保存：KF state + covariance P + age + event flag', 4.08, 3.75, 5.20, C.blue, C.panel2);
  const temporalCases = [
    ['信息稳定', '过程噪声 Q 较小', '连续一致的观测逐步减小协方差，方向判断不易抖动。', C.blue],
    ['新证据可信', '观测噪声 R 较小', 'Kalman gain 增大，策略更快跟随可靠的新潜力判断。', C.green],
    ['结构发生变化', '提高 Q 或直接重置', 'frontier 消失、边失效时迅速放弃已经过时的 belief。', C.orange],
  ];
  temporalCases.forEach((item, i) => {
    const x = 0.72 + i * 3.95;
    rect(s, x, 4.10, 3.68, 1.62, i === 1 ? 'E3F4EB' : C.panel, 0.08, item[3]);
    addText(s, item[0], x + 0.24, 4.35, 1.28, 0.26, { fontSize: 11.8, color: item[3], bold: true });
    addText(s, item[1], x + 1.58, 4.35, 1.72, 0.26, { fontSize: 9.2, color: C.text, bold: true, align: 'right' });
    addText(s, item[2], x + 0.24, 4.82, 3.18, 0.54, { fontSize: 9.5, color: C.muted, breakLine: true });
  });
  rect(s, 0.72, 6.02, 11.58, 0.40, 'FFF3D8', 0.05, C.orange);
  addText(s, '保留 Kalman Filter，但滤的是未来潜力 belief；raw utility 仍直接更新，不让滤波延迟掩盖真实 frontier 事件。', 0.98, 6.12, 11.06, 0.20, { fontSize: 9.4, color: C.orange, bold: true, align: 'center' });
  source(s, 'Event-gated Adaptive Kalman Filter on latent future potential, proposed');
  note(s, '算法 B 是自适应 Kalman Filter，而不是原先的 raw Utility KF。状态 x 是节点未来潜力，Potential Head 输出是观测 z，预测方差映射为观测噪声 R；地图稳定时过程噪声 Q 较小，连续观测会降低 posterior covariance。检测到节点访问、frontier 消失、边失效或区域合并分裂时，提高 Q 使滤波器快速跟随，或直接重置状态。这样既保留 Kalman 的不确定性递推，又不会把真实结构突变当作普通噪声抹平。');
}

// 15. Multi-scale graph
{
  const s = pptx.addSlide('MASTER');
  img(s, asset('upbg_value_aware_multiscale_graph.png'), 0, 0, 13.333, 7.50);
  note(s, '算法 C 采用 Random-Walk Graph Wavelet Transform。对节点隐藏特征分别做 1/2/4-hop 随机游走平滑，低频 LF 表示房间、长走廊和区域趋势，高频 HF 等于原始特征减低频，突出 frontier、拐角和拓扑变化。远端 LF 特征输入 DiffPool 形成区域 token；机器人附近的高频节点按未来潜力、预测方差和 wavelet energy 做 Top-K 保留。最后用 A* 最短路径骨架补回连接区域 token 与局部动作图所需的节点。Encoder 用 LF 构造全局 Q/K，Value 分支保留原始特征与 HF residual；Pointer Decoder 最终仍只选择当前可行邻居。');
}

// 16. Training and deployment
{
  const s = pptx.addSlide('MASTER');
  title(s, '15 / Training strategy', '完整地图只负责教学，机器人实际运行时只看局部地图', '训练和部署分成两条独立通道，避免把“训练时知道答案”误解为“运行时偷看答案”。');
  rect(s, 0.72, 2.04, 11.88, 1.74, 'F1EDF9', 0.08, C.purple);
  pill(s, '训练阶段', 0.98, 2.32, 1.08, C.purple, 'FFFFFF');
  const trainX = [2.24, 5.62, 9.00];
  const trainSteps = [
    ['GT Teacher 生成标签与 LF/HF', 'future-gain rollout + Graph Wavelet 教师特征'],
    ['Student 联合学习', '蒸馏 + Gaussian NLL + RankNet + SAC'],
    ['训练完成后移除教师', '完整地图和教师网络都不进入部署'],
  ];
  trainSteps.forEach((item, i) => {
    node(s, trainX[i], 2.45, item[0], 'FFFFFF', C.purple, 2.62, 0.62, 9.4);
    addText(s, item[1], trainX[i], 3.18, 2.62, 0.27, { fontSize: 8.7, color: C.muted, align: 'center' });
    if (i < trainSteps.length - 1) arrow(s, trainX[i] + 2.72, 2.76, trainX[i + 1] - 0.10, 2.76, C.purple, 1.1, 'dash');
  });
  rect(s, 0.72, 4.18, 11.88, 1.74, C.panel2, 0.08, C.blue2);
  pill(s, '部署阶段', 0.98, 4.46, 1.08, C.blue, 'FFFFFF');
  const deployX = [2.24, 5.62, 9.00];
  const deploySteps = [
    ['输入局部地图', '只使用机器人当下真实可见的信息'],
    ['KF + Graph Wavelet 更新', '维护潜力 belief 并分解远近尺度'],
    ['Pointer 输出下一航点', '在候选邻居中选择下一步'],
  ];
  deploySteps.forEach((item, i) => {
    node(s, deployX[i], 4.59, item[0], 'FFFFFF', [C.blue, C.green, C.orange][i], 2.62, 0.62, 9.4);
    addText(s, item[1], deployX[i], 5.32, 2.62, 0.27, { fontSize: 8.7, color: C.muted, align: 'center' });
    if (i < deploySteps.length - 1) arrow(s, deployX[i] + 2.72, 4.90, deployX[i + 1] - 0.10, 4.90, C.blue2, 1.1);
  });
  addText(s, '一句话：完整地图是老师，不是机器人的额外传感器。', 0.98, 6.25, 11.32, 0.24, { fontSize: 10.5, color: C.blue, bold: true, align: 'center' });
  source(s, 'No privileged information at deployment');
  note(s, '训练期 GT Teacher 提供 future-gain 标签和 Graph Wavelet 的 LF/HF 教师特征，Student 通过特权蒸馏、Gaussian NLL、RankNet 与 SAC 联合训练。部署期教师和 GT map 全部移除，只保留 Student Potential Head、自适应 KF、Graph Wavelet/DiffPool 与 Pointer Actor。');
}

// 17. Decision story
{
  const s = pptx.addSlide('MASTER');
  title(s, '16 / Decision story', '机器人走到一个路口时，整套方法怎样改变行为', '把三个改进放回一次真实决策：它们不是并排工作的模块，而是连续完成同一次判断。');
  const storyX = [0.68, 3.15, 5.62, 8.09, 10.56];
  const story = [
    ['01', '看见路口', '局部地图给出 A、B 两个可行方向。', C.blue2],
    ['02', '异方差预测', 'Potential Head 判断 B 更可能连到新房间。', C.blue],
    ['03', 'Kalman 更新', '连续观测支持 B，且没有结构事件触发重置。', C.green],
    ['04', 'Wavelet 重组', '低频保留 B 的区域趋势，高频保留去 B 的局部路径。', C.orange],
    ['05', '选择 B', '策略愿意接受适度路程，换取更大的长期发现。', C.yellow],
  ];
  story.forEach((item, i) => {
    rect(s, storyX[i], 2.18, 2.10, 2.70, C.panel, 0.08, item[3]);
    pill(s, item[0], storyX[i] + 0.22, 2.43, 0.62, item[3], i === 3 ? 'FFF3D8' : C.panel2);
    addText(s, item[1], storyX[i] + 0.22, 2.94, 1.64, 0.30, { fontSize: 12.4, color: item[3], bold: true, align: 'center' });
    addText(s, item[2], storyX[i] + 0.22, 3.45, 1.66, 0.92, { fontSize: 9.4, color: C.muted, breakLine: true, align: 'center', valign: 'top' });
    if (i < story.length - 1) arrow(s, storyX[i] + 2.18, 3.53, storyX[i + 1] - 0.08, 3.53, C.blue2, 1.1);
  });
  rect(s, 0.72, 5.26, 5.70, 0.82, 'FFF3D8', 0.06, C.orange);
  addText(s, '只看眼前', 0.98, 5.46, 1.12, 0.24, { fontSize: 10.5, color: C.orange, bold: true });
  addText(s, '容易被 A 的高 frontier 吸引，进入尽头后再回退。', 2.20, 5.41, 3.92, 0.36, { fontSize: 9.7, color: C.muted, breakLine: true });
  rect(s, 6.68, 5.26, 5.70, 0.82, 'E3F4EB', 0.06, C.green);
  addText(s, '最终方法', 6.94, 5.46, 1.12, 0.24, { fontSize: 10.5, color: C.green, bold: true });
  addText(s, '识别 B 的长期潜力并验证可信度，更早作出非短视选择。', 8.16, 5.41, 3.92, 0.36, { fontSize: 9.7, color: C.muted, breakLine: true });
  source(s, 'Behavioral explanation of the unified method');
  note(s, '这一页是最“人话”的方法总结。按路口场景顺序讲：看到候选方向、判断后续、核对可信度、保留关键结构、最终选择。');
}

// 18. Comparison
{
  const s = pptx.addSlide('MASTER');
  title(s, '17 / Method comparison', '相对 baseline，真正改变的是“如何理解未知区域”', '动作空间和强化学习主干可以保留；优化重点落在节点评价、时间记忆和空间组织方式。');
  const x0 = 0.72;
  const widths = [1.90, 2.92, 3.40, 3.58];
  tableHeader(s, ['比较维度', 'Cao baseline', '最终优化方法', '预期行为变化'], x0, 2.02, widths);
  const rows = [
    ['节点评价', '重点使用当前 utility，长期价值主要由策略隐式学习', '特权蒸馏 + 异方差 Potential Head + RankNet', '不再只追逐眼前最显眼的 frontier'],
    ['时间信息', '每轮主要依据当前图快照重新决策', '结构事件门控的 Adaptive Kalman Filter', '减少方向反复切换和过时判断'],
    ['空间尺度', '统一节点图配合固定规则控制规模', 'Random-Walk Graph Wavelet + DiffPool / Top-K', '地图变大后仍能兼顾远程方向与局部执行'],
    ['训练信息', '完整地图主要帮助 critic 更稳定地估值', 'GT Teacher 蒸馏 future-gain 与 LF/HF 特征', '训练获得更密集提示，部署输入保持不变'],
    ['动作取舍', '由当前图表征和价值函数综合决定', 'LF-guided Attention + HF residual + Pointer', '决策理由更清楚，也更容易做针对性消融'],
  ];
  rows.forEach((row, ri) => {
    let x = x0;
    const y = 2.50 + ri * 0.72;
    row.forEach((value, ci) => {
      rect(s, x, y, widths[ci], 0.72, ri % 2 === 0 ? C.panel : C.panel2, 0.01, C.line);
      addText(s, value, x + 0.12, y + 0.09, widths[ci] - 0.24, 0.52, { fontSize: 8.9, color: ci === 2 ? C.teal : (ci === 0 ? C.text : C.muted), bold: ci === 0 || ci === 2, breakLine: true, valign: 'mid' });
      x += widths[ci];
    });
  });
  rect(s, 0.72, 6.18, 11.80, 0.34, C.panel2, 0.05, C.blue2);
  addText(s, '不改变任务定义：仍是在局部地图上选择可行邻居作为下一导航点。', 0.98, 6.27, 11.28, 0.17, { fontSize: 9.5, color: C.blue, bold: true, align: 'center' });
  source(s, 'Conceptual comparison; proposed items await experiments');
  note(s, '只比较 baseline 和最终方法。强调保留动作空间和训练主干，真正改变的是对未知区域的显式建模方式。');
}

// 19. Contributions
{
  const s = pptx.addSlide('MASTER');
  title(s, '18 / Contributions', '论文贡献可以收敛为三点，共同回答一个问题', '问题只有一个：怎样让机器人在只看见局部地图时，对未知区域作出更前瞻、更可信的判断。');
  const contrib = [
    ['贡献 1', '异方差潜力蒸馏', 'GT rollout 构造 future-gain，Teacher–Student 蒸馏与 Gaussian NLL 学习潜力和预测方差，RankNet 学邻居排序。', '验证重点：潜力误差、校准误差与候选排序质量', C.blue],
    ['贡献 2', '事件门控自适应 KF', '以未来潜力为状态、Potential Head 为观测；用预测方差设置 R，用结构事件调 Q 或重置 posterior。', '验证重点：方向抖动、事件响应时间与 KF 消融', C.green],
    ['贡献 3', 'Graph Wavelet 多尺度图', '1/2/4-hop 图小波分离 LF/HF；DiffPool 形成区域 token，Top-K 与 A* 骨架保留高频细节和连通。', '验证重点：尺度扩展、节点预算和 LF/HF 消融', C.orange],
  ];
  contrib.forEach((item, i) => {
    const x = 0.72 + i * 3.89;
    rect(s, x, 2.10, 3.56, 3.58, C.panel, 0.08, item[4]);
    pill(s, item[0], x + 0.24, 2.38, 0.92, item[4], i === 2 ? 'FFF3D8' : C.panel2);
    addText(s, item[1], x + 0.24, 2.90, 3.04, 0.30, { fontSize: 13.2, color: item[4], bold: true });
    addText(s, item[2], x + 0.24, 3.42, 3.04, 1.10, { fontSize: 9.7, color: C.muted, breakLine: true, valign: 'top' });
    line(s, x + 0.24, 4.82, x + 3.28, 4.82, C.line, 0.7);
    addText(s, item[3], x + 0.24, 5.02, 3.04, 0.44, { fontSize: 8.9, color: item[4], bold: true, breakLine: true });
  });
  rect(s, 0.72, 5.96, 11.80, 0.48, 'E3F4EB', 0.06, C.green);
  addText(s, '统一假设：当未来潜力判断更准确、时间上更稳定、空间上更完整时，策略会更少短视、回退和重复运动。', 0.98, 6.09, 11.28, 0.22, { fontSize: 10, color: C.teal, bold: true, align: 'center' });
  source(s, 'Proposed paper contribution statement');
  note(s, '贡献不要写成模块清单。三点分别解决“看多远、信多少、怎样组织远近信息”，最后都落到同一个行为假设。');
}

// 20. Experiment plan
{
  const s = pptx.addSlide('MASTER');
  title(s, '19 / Validation plan', '不是只比较最终模型，而是逐层回答每项改进是否有用', '消融顺序与论文论点一一对应；任何一项没有独立证据，就不能把它写成核心贡献。');
  rect(s, 0.72, 2.04, 5.64, 4.08, C.panel, 0.08, C.line);
  addText(s, '逐层消融', 1.02, 2.32, 1.60, 0.28, { fontSize: 13.5, color: C.text, bold: true });
  const ablations = [
    ['1', 'Cao baseline', '确定原始参照线', C.blue2],
    ['2', 'Potential Head + RankNet', '验证 future-gain 与排序是否学得出来', C.blue],
    ['3', '+ Adaptive Kalman Filter', '验证稳定性与结构事件响应', C.green],
    ['4', '+ Graph Wavelet / DiffPool', '验证大地图下的 LF/HF 与节点预算', C.orange],
    ['5', '完整方法', '检查三项机制能否形成互补', C.purple],
  ];
  ablations.forEach((item, i) => {
    const y = 2.90 + i * 0.57;
    circle(s, 1.04, y + 0.01, 0.26, item[3], 'none');
    addText(s, item[0], 1.04, y + 0.085, 0.26, 0.10, { fontFace: A, fontSize: 7.4, color: 'FFFFFF', bold: true, align: 'center' });
    addText(s, item[1], 1.48, y, 2.02, 0.24, { fontSize: 9.8, color: item[3], bold: true });
    addText(s, item[2], 3.56, y, 2.35, 0.30, { fontSize: 8.8, color: C.muted, breakLine: true });
    if (i < ablations.length - 1) line(s, 1.17, y + 0.30, 1.17, y + 0.55, C.line, 0.8);
  });
  rect(s, 6.62, 2.04, 5.98, 4.08, C.panel, 0.08, C.line);
  addText(s, '重点观察什么', 6.92, 2.32, 2.10, 0.28, { fontSize: 13.5, color: C.text, bold: true });
  const observations = [
    ['预测是否靠谱', '高潜力节点能否稳定排在前面；可信度是否与误差相符', C.blue],
    ['路径是否更省', '总路程、回退次数、重复经过的路段是否减少', C.green],
    ['行为是否更稳', '相邻方向来回切换、死胡同恢复和结构变化响应', C.orange],
    ['地图变大是否扛得住', '节点数量、规划耗时、内存和路径效率随规模的变化', C.purple],
    ['新环境是否还能用', '未见过的室内布局、室外结构和传感器噪声', C.red],
  ];
  observations.forEach((item, i) => {
    const y = 2.90 + i * 0.57;
    addText(s, item[0], 6.94, y, 1.54, 0.24, { fontSize: 9.6, color: item[2], bold: true });
    addText(s, item[1], 8.58, y, 3.44, 0.31, { fontSize: 8.8, color: C.muted, breakLine: true });
    if (i < observations.length - 1) line(s, 6.94, y + 0.40, 12.05, y + 0.40, C.line, 0.5);
  });
  rect(s, 0.72, 6.30, 11.88, 0.34, C.panel2, 0.05, C.blue2);
  addText(s, '公平比较：地图划分、训练预算、随机种子、传感器范围、动作空间和模型选择规则保持一致。', 0.98, 6.39, 11.36, 0.17, { fontSize: 9.4, color: C.blue, bold: true, align: 'center' });
  source(s, 'Planned ablation and evaluation protocol');
  note(s, '讲消融时先说“为什么按这个顺序”。未来潜力是前提；时序与多尺度分别验证；最后才看完整方法是否互补。');
}

// 21. Expected outcomes
{
  const s = pptx.addSlide('MASTER');
  title(s, '20 / Expected outcomes', '预期效果要落到机器人行为，而不是先写一个提升百分比', '目前所有结论都是待验证假设；实验完成后再填真实均值、波动范围和显著性。');
  tableHeader(s, ['预期变化', '机器人会表现成什么样', '怎样判断是否真的发生'], 0.74, 2.02, [2.18, 5.18, 4.50]);
  const outcomes = [
    ['更前瞻', '不容易被眼前 frontier 多但后续贫乏的方向吸引；更早选择能打开新区域的入口。', '回退、重复路段和进入死胡同后的补救路程减少', C.blue],
    ['更稳定', '单帧预测抖动时不频繁换方向；通路或 frontier 真正变化时又能及时改判。', '方向切换次数下降，结构变化后的恢复更快', C.green],
    ['更适合大地图', '远处有值得去的区域时不会因图压缩被完全丢掉，近处动作仍然可执行。', '地图扩大后路径效率下降更慢，规划开销保持可控', C.orange],
    ['更容易泛化', '遇到训练中少见的布局时，仍能依靠“后续可能打开什么”作出合理选择。', '新布局和室外场景中的成功率、路径表现更稳定', C.purple],
  ];
  outcomes.forEach((item, i) => {
    const y = 2.50 + i * 0.90;
    const widths = [2.18, 5.18, 4.50];
    let x = 0.74;
    [item[0], item[1], item[2]].forEach((value, ci) => {
      rect(s, x, y, widths[ci], 0.90, i % 2 === 0 ? C.panel : C.panel2, 0.01, C.line);
      addText(s, value, x + 0.14, y + 0.12, widths[ci] - 0.28, 0.64, { fontSize: ci === 0 ? 10.8 : 9.2, color: ci === 0 ? item[3] : C.muted, bold: ci === 0, breakLine: true, valign: 'mid', align: ci === 0 ? 'center' : 'left' });
      x += widths[ci];
    });
  });
  rect(s, 0.74, 6.28, 11.86, 0.34, 'FFF3D8', 0.05, C.orange);
  addText(s, '关键验收：提升应来自更好的未知区域判断，而不是更大网络、额外部署信息或不公平训练预算。', 1.00, 6.37, 11.34, 0.17, { fontSize: 9.4, color: C.orange, bold: true, align: 'center' });
  source(s, 'Expected outcomes are hypotheses, not completed findings');
  note(s, '这一页避免任何虚构数字。每项预期都写成“机器人会怎样表现”和“用什么现象判断”，方便后续直接替换成实验结果。');
}

// 22. Roadmap
{
  const s = pptx.addSlide('MASTER');
  title(s, '21 / Roadmap', '按科学风险推进：先证明“未来潜力”值得学，再扩展完整方法', '每一阶段都设置停止条件，避免在核心假设尚未成立时继续堆叠复杂度。');
  const stages = [
    ['阶段 1', '定义教学答案', '明确“从候选点继续走能发现什么”，抽查典型房间、走廊和死胡同案例。', '产出：标签分布与可视化', C.blue],
    ['阶段 2', '蒸馏 Potential Head', '训练 Gaussian NLL 与 RankNet，先验证潜力、预测方差和候选排序。', '产出：单项消融结果', C.green],
    ['阶段 3', '加入 KF 与 Wavelet', '分别测试 Adaptive KF、Graph Wavelet 和 DiffPool，定位各自作用。', '产出：两组独立消融', C.orange],
    ['阶段 4', '完成论文闭环', '运行完整方法、规模测试与新环境测试，补充复杂度、失败案例和统计检验。', '产出：论文主实验与初稿', C.purple],
  ];
  stages.forEach((item, i) => {
    const x = 0.78 + i * 3.07;
    rect(s, x, 2.26, 2.70, 3.18, C.panel, 0.08, item[4]);
    pill(s, item[0], x + 0.24, 2.54, 0.88, item[4], i === 2 ? 'FFF3D8' : C.panel2);
    addText(s, item[1], x + 0.24, 3.04, 2.20, 0.30, { fontSize: 12.5, color: item[4], bold: true });
    line(s, x + 0.24, 3.52, x + 2.46, 3.52, C.line, 0.7);
    addText(s, item[2], x + 0.24, 3.76, 2.22, 0.92, { fontSize: 9.2, color: C.muted, breakLine: true, valign: 'top' });
    addText(s, item[3], x + 0.24, 4.92, 2.22, 0.24, { fontSize: 8.7, color: item[4], bold: true, align: 'center' });
    if (i < stages.length - 1) arrow(s, x + 2.77, 3.78, x + 3.00, 3.78, C.blue2, 1.0);
  });
  rect(s, 0.78, 5.78, 11.91, 0.54, C.panel2, 0.06, C.blue2);
  addText(s, '关键决策点：如果未来潜力预测不能稳定改善节点排序，就先修正教学答案与监督方式，不继续增加时序和多尺度复杂度。', 1.04, 5.92, 11.38, 0.26, { fontSize: 9.8, color: C.blue, bold: true, align: 'center' });
  source(s, 'Risk-ordered research roadmap');
  note(s, '路线图按科学问题排序。第一风险不是工程集成，而是未来潜力标签是否有信息量、模型是否学得会。只有这一点成立，后续机制才有意义。');
}

// 23. References
{
  const s = pptx.addSlide('MASTER');
  title(s, '22 / References', '论文依据与具体算法落点', 'Baseline 界定任务主干；蒸馏、异方差回归、Kalman Filter 与 Graph Wavelet 支撑三项改进。');
  const refs = [
    ['[1]', 'Cao et al., ICRA 2023', 'ARiADNE：图注意力策略、离散 SAC 与非短视探索；论文展望提出潜在收益预测。', C.blue],
    ['[2]', 'Cao et al., IEEE RA-L 2024', '大规模探索：privileged critic、图稀疏化、小场景到大场景迁移及真实机器人验证。', C.green],
    ['[3]', 'Kendall & Gal, NeurIPS 2017; Burges et al., 2005', '异方差 Gaussian NLL 学预测方差；RankNet 式成对排序直接监督候选节点顺序。', C.cyan],
    ['[4]', 'Kalman, ASME 1960; Mehra, IEEE TAC 1970', 'Kalman Filter 递推状态与协方差；自适应噪声估计支持根据预测方差和结构事件调整 R 与 Q。', C.orange],
    ['[5]', 'Hammond et al., 2011; Ying et al., 2018; Gao & Ji, 2019', 'Graph Wavelet 分离多尺度信号；DiffPool 形成区域 token；Graph U-Nets 提供 Top-K pooling 依据。', C.purple],
    ['[6]', 'Hinton et al., 2015', '特权信息蒸馏：训练期 GT Teacher 提供 future-gain 与 LF/HF 教师特征，部署只保留 Student。', C.yellow],
  ];
  refs.forEach((item, i) => {
    const y = 2.02 + i * 0.66;
    rect(s, 0.74, y, 11.86, 0.54, i % 2 === 0 ? C.panel : C.panel2, 0.04, C.line);
    addText(s, item[0], 0.98, y + 0.15, 0.42, 0.18, { fontFace: A, fontSize: 10, color: item[3], bold: true });
    addText(s, item[1], 1.56, y + 0.10, 3.02, 0.22, { fontFace: A, fontSize: 9.3, color: C.text, bold: true });
    addText(s, item[2], 4.76, y + 0.08, 7.44, 0.34, { fontSize: 8.8, color: C.muted, breakLine: true });
  });
  rect(s, 0.74, 6.18, 11.86, 0.40, 'FFF3D8', 0.05, C.orange);
  addText(s, '边界说明：baseline 数字来自已发表论文；UPBG-RL 页面描述的是研究计划，尚未宣称实验提升。', 1.00, 6.28, 11.34, 0.20, { fontSize: 9.5, color: C.orange, bold: true, align: 'center' });
  source(s, 'Selected references for the proposed paper direction');
  note(s, '引用页只保留论文与理论依据。正式写作时再扩展主动探索、图规划、不确定性校准和特权学习相关工作。');
}

// 24. Closing
{
  const s = pptx.addSlide('MASTER');
  s.background = { color: C.bg };
  addText(s, 'CONCLUSION', 0.72, 0.78, 3.0, 0.25, { fontFace: A, fontSize: 10, color: C.blue, bold: true, charSpacing: 1.4 });
  rect(s, 0, 1.36, 13.333, 1.34, C.blue, 0, C.blue);
  addText(s, '从“眼前哪里有 frontier”，升级到“哪条路更可能打开新空间”。', 0.72, 1.68, 11.90, 0.52, { fontSize: 22, color: 'FFFFFF', bold: true, align: 'center' });
  addText(s, 'UPBG-RL', 0.72, 2.28, 11.90, 0.24, { fontFace: A, fontSize: 11, color: 'DBF5F9', bold: true, align: 'center' });
  line(s, 0.76, 3.12, 12.45, 3.12, C.line, 0.8);
  const final = [
    ['看得更远', '预测候选方向后面可能打开的新空间，不只比较当前 frontier。', C.blue],
    ['判断更可信', '连续积累证据；预测不可靠时谨慎，结构变化时及时重估。', C.green],
    ['远近都兼顾', '区域级方向负责长期规划，局部 frontier 和通路负责下一步执行。', C.orange],
  ];
  final.forEach((item, i) => {
    const x = 0.78 + i * 4.02;
    rect(s, x, 3.46, 3.48, 1.62, C.panel, 0.08, item[2]);
    circle(s, x + 0.24, 3.75, 0.20, item[2], 'none');
    addText(s, item[0], x + 0.58, 3.68, 2.46, 0.26, { fontSize: 12.8, color: item[2], bold: true });
    addText(s, item[1], x + 0.24, 4.16, 2.98, 0.56, { fontSize: 9.7, color: C.muted, breakLine: true });
  });
  addText(s, '一句话总结', 0.78, 5.66, 1.30, 0.30, { fontSize: 14, color: C.blue, bold: true });
  addText(s, '最终方法把未来潜力、时序可信度和多尺度图组织成同一条决策链，目标是减少短视选择、回退和重复运动。', 2.12, 5.61, 9.45, 0.58, { fontSize: 11, color: C.text, bold: true, breakLine: true });
  addText(s, '谢谢 / Q&A', 0.78, 6.42, 3.0, 0.30, { fontFace: A, fontSize: 15, color: C.blue, bold: true });
  note(s, '结束时回到最直白的一句话：机器人不仅看眼前哪里信息多，还要判断哪条路更可能真正打开新空间，并知道自己是否判断得准。');
}

pptx.writeFile({ fileName: path.join(__dirname, 'KF-Enhanced-DRL-Exploration-论文改进计划.pptx') });
