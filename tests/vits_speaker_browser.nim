## vits_speaker_browser.nim
## VITS TTS 说话人浏览器工具
## 为所有说话人生成测试音频并生成 HTML 浏览页面

import std/[os, strutils, tables, sequtils, times, strformat]
import onnxruntime, vits_utils

# ==================== 配置区域 ====================
const
  # 模型路径配置
  TestDataDir = "tests/testdata/vits-icefall-zh-aishell3"
  ModelPath = TestDataDir / "model.onnx"
  LexiconPath = TestDataDir / "lexicon.txt"
  TokensPath = TestDataDir / "tokens.txt"
  
  # === 输出目录配置 - 可修改 ===
  OutputDir = "./speaker_samples"  # 改为相对路径或绝对路径，如 "/Users/bung/audio_output"
  
  # 音频参数
  SampleRate = 8000  # AISHELL3 使用 8kHz
  TotalSpeakers = 174  # 总说话人数量

# 测试句子配置
const TestSentences = [
  ("数字测试", "一二三四五六七八九十"),
  ("问候语", "你好，很高兴认识你。"),
  ("自我介绍", "我是人工智能助手，可以帮你完成各种任务。"),
  ("天气描述", "今天天气晴朗，阳光明媚，非常适合户外活动。"),
  ("诗歌朗诵", "床前明月光，疑是地上霜。举头望明月，低头思故乡。"),
  ("新闻播报", "据新华社报道，我国经济持续向好发展，人民生活水平不断提高。"),
  ("情感表达", "谢谢你一直以来的支持和陪伴，我真的很感激。"),
  ("技术说明", "语音合成技术可以将文字转换为自然流畅的语音。"),
  ("长句测试", "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾。"),
]

# ==================== HTML 模板 ====================
const HtmlTemplate = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VITS AISHELL3 说话人浏览器</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container { max-width: 1400px; margin: 0 auto; }
        
        header {
            text-align: center;
            color: white;
            padding: 40px 20px;
            margin-bottom: 30px;
        }
        
        header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        header p { font-size: 1.1rem; opacity: 0.9; }
        
        .controls {
            background: white;
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }
        
        .control-group { margin-bottom: 20px; }
        
        .control-group label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }
        
        .sentence-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .sentence-btn {
            padding: 8px 16px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s ease;
        }
        
        .sentence-btn:hover, .sentence-btn.active {
            background: #667eea;
            color: white;
        }
        
        .filter-controls {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }
        
        input[type="text"], input[type="number"] {
            padding: 10px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        input[type="text"]:focus, input[type="number"]:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .stats {
            display: flex;
            gap: 30px;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #eee;
            color: #666;
            font-size: 14px;
            flex-wrap: wrap;
        }
        
        .stat-item strong { color: #333; font-size: 1.2em; }
        
        .speakers-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 20px;
        }
        
        .speaker-card {
            background: white;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .speaker-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
        }
        
        .speaker-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .speaker-id { font-size: 1.3rem; font-weight: bold; }
        .speaker-badge { background: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 12px; font-size: 12px; }
        
        .speaker-body { padding: 20px; }
        
        .audio-item {
            margin-bottom: 15px;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .audio-label { font-size: 12px; color: #666; margin-bottom: 8px; font-weight: 500; }
        .audio-text { font-size: 13px; color: #333; margin-bottom: 8px; line-height: 1.4; }
        
        audio {
            width: 100%;
            height: 36px;
            border-radius: 18px;
        }
        
        .action-buttons {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        
        .action-btn {
            flex: 1;
            padding: 8px 12px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.3s;
        }
        
        .btn-compare {
            background: #17a2b8;
            color: white;
        }
        
        .btn-compare:hover { background: #138496; }
        .btn-compare.active { background: #dc3545; }
        
        .hidden { display: none !important; }
        
        footer {
            text-align: center;
            color: white;
            padding: 40px;
            opacity: 0.8;
        }
        
        /* 对比面板 */
        .compare-panel {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: white;
            padding: 20px;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            width: 350px;
            max-height: 400px;
            display: none;
            flex-direction: column;
            z-index: 1000;
        }
        
        .compare-panel.visible { display: flex; }
        
        .compare-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        
        .compare-header h3 { color: #333; font-size: 16px; }
        
        .remove-btn {
            background: #dc3545;
            color: white;
            border: none;
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        
        .compare-list { overflow-y: auto; flex: 1; margin-bottom: 15px; }
        
        .compare-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px;
            background: #f8f9fa;
            margin-bottom: 5px;
            border-radius: 6px;
            font-size: 14px;
        }
        
        .compare-actions { display: flex; gap: 10px; }
        
        .compare-actions button {
            flex: 1;
            padding: 10px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .btn-primary { background: #667eea; color: white; }
        .btn-secondary { background: #6c757d; color: white; }
        
        @media (max-width: 768px) {
            .speakers-grid { grid-template-columns: 1fr; }
            header h1 { font-size: 1.8rem; }
            .compare-panel { left: 10px; right: 10px; width: auto; }
            .filter-controls { flex-direction: column; align-items: stretch; }
            .filter-controls input { width: 100%; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎙️ VITS AISHELL3 说话人浏览器</h1>
            <p>浏览和比较 ${TOTAL_SPEAKERS} 个说话人的语音样本</p>
        </header>
        
        <div class="controls">
            <div class="control-group">
                <label>📝 选择测试句子：</label>
                <div class="sentence-buttons" id="sentenceButtons"></div>
            </div>
            
            <div class="control-group">
                <label>🔍 筛选说话人：</label>
                <div class="filter-controls">
                    <input type="text" id="searchInput" placeholder="搜索 ID (如 10, 20-30)">
                    <input type="number" id="rangeStart" placeholder="起始 ID" min="0" max="${MAX_SID}">
                    <span style="color: #666;">-</span>
                    <input type="number" id="rangeEnd" placeholder="结束 ID" min="0" max="${MAX_SID}">
                    <button class="sentence-btn" onclick="applyFilter()">应用</button>
                    <button class="sentence-btn" onclick="resetFilter()">重置</button>
                </div>
            </div>
            
            <div class="stats">
                <div class="stat-item">总说话人：<strong>${TOTAL_SPEAKERS}</strong></div>
                <div class="stat-item">当前显示：<strong id="visibleCount">0</strong></div>
                <div class="stat-item">测试句子：<strong>${TOTAL_SENTENCES}</strong> 条</div>
                <div class="stat-item">对比列表：<strong id="compareCount">0</strong></div>
            </div>
        </div>
        
        <div class="speakers-grid" id="speakersGrid"></div>
        
        <footer>
            <p>Generated by VITS Speaker Browser | ONNX Runtime Nim Binding</p>
        </footer>
    </div>
    
    <div class="compare-panel" id="comparePanel">
        <div class="compare-header">
            <h3>🔄 对比列表</h3>
            <button class="remove-btn" onclick="clearCompare()">清空</button>
        </div>
        <div class="compare-list" id="compareList"></div>
        <div class="compare-actions">
            <button class="btn-primary" onclick="playCompare()">▶ 顺序播放</button>
            <button class="btn-secondary" onclick="exportList()">📥 导出</button>
        </div>
    </div>

    <script>
        // 配置
        const TOTAL_SPEAKERS = ${TOTAL_SPEAKERS};
        const SENTENCES = ${SENTENCES_JS_ARRAY};
        const SENTENCE_TEXTS = ${SENTENCES_TEXT_JS};
        
        // 状态
        let currentSentence = SENTENCES[0];
        let compareList = [];
        let filteredSpeakers = [];
        
        // 构造音频文件名 (格式: speaker_XXX_句子名.wav)
        function getAudioUrl(sid, sentence) {
            const sidStr = sid.toString().padStart(3, '0');
            return `speaker_${sidStr}_${sentence}.wav`;
        }
        
        // 初始化
        function init() {
            renderSentenceButtons();
            filterSpeakers();
        }
        
        // 渲染句子按钮
        function renderSentenceButtons() {
            const container = document.getElementById('sentenceButtons');
            container.innerHTML = SENTENCES.map(s => `
                <button class="sentence-btn ${s === currentSentence ? 'active' : ''}" 
                        onclick="switchSentence('${s}')">${s}</button>
            `).join('');
        }
        
        // 切换句子
        function switchSentence(sentence) {
            currentSentence = sentence;
            renderSentenceButtons();
            renderSpeakers();
        }
        
        // 筛选说话人
        function filterSpeakers() {
            const searchVal = document.getElementById('searchInput').value.trim();
            const rangeStart = parseInt(document.getElementById('rangeStart').value) || 0;
            const rangeEnd = parseInt(document.getElementById('rangeEnd').value) || (TOTAL_SPEAKERS - 1);
            
            filteredSpeakers = [];
            
            // 解析搜索值 (支持: 10, 20, 30-40)
            let searchIds = new Set();
            if (searchVal) {
                searchVal.split(/[,，]/).forEach(part => {
                    part = part.trim();
                    if (part.includes('-')) {
                        const [start, end] = part.split('-').map(x => parseInt(x.trim()));
                        if (!isNaN(start) && !isNaN(end)) {
                            for (let i = start; i <= end; i++) searchIds.add(i);
                        }
                    } else {
                        const id = parseInt(part);
                        if (!isNaN(id)) searchIds.add(id);
                    }
                });
            }
            
            // 筛选
            for (let sid = 0; sid < TOTAL_SPEAKERS; sid++) {
                let visible = true;
                
                // 搜索筛选
                if (searchIds.size > 0 && !searchIds.has(sid)) {
                    visible = false;
                }
                
                // 范围筛选
                if (sid < rangeStart || sid > rangeEnd) {
                    visible = false;
                }
                
                if (visible) filteredSpeakers.push(sid);
            }
            
            document.getElementById('visibleCount').textContent = filteredSpeakers.length;
            renderSpeakers();
        }
        
        function applyFilter() { filterSpeakers(); }
        
        function resetFilter() {
            document.getElementById('searchInput').value = '';
            document.getElementById('rangeStart').value = '';
            document.getElementById('rangeEnd').value = '';
            filterSpeakers();
        }
        
        // 渲染说话人卡片
        function renderSpeakers() {
            const grid = document.getElementById('speakersGrid');
            grid.innerHTML = filteredSpeakers.map(sid => `
                <div class="speaker-card" data-speaker-id="${sid}">
                    <div class="speaker-header">
                        <span class="speaker-id">说话人 ${sid}</span>
                        <span class="speaker-badge">ID: ${sid}</span>
                    </div>
                    <div class="speaker-body">
                        <div class="audio-item">
                            <div class="audio-label">${currentSentence}</div>
                            <div class="audio-text">${SENTENCE_TEXTS[currentSentence]}</div>
                            <audio controls id="audio-${sid}">
                                <source src="${getAudioUrl(sid, currentSentence)}" type="audio/wav">
                                浏览器不支持音频播放
                            </audio>
                        </div>
                        <div class="action-buttons">
                            <button class="action-btn btn-compare ${compareList.includes(sid) ? 'active' : ''}" 
                                    onclick="toggleCompare(${sid})">
                                ${compareList.includes(sid) ? '✓ 已加入' : '+ 对比'}
                            </button>
                        </div>
                    </div>
                </div>
            `).join('');
        }
        
        // 对比功能
        function toggleCompare(sid) {
            const idx = compareList.indexOf(sid);
            if (idx === -1) {
                compareList.push(sid);
            } else {
                compareList.splice(idx, 1);
            }
            updateComparePanel();
            renderSpeakers();
        }
        
        function updateComparePanel() {
            const panel = document.getElementById('comparePanel');
            const listDiv = document.getElementById('compareList');
            
            document.getElementById('compareCount').textContent = compareList.length;
            
            if (compareList.length === 0) {
                panel.classList.remove('visible');
                return;
            }
            
            panel.classList.add('visible');
            listDiv.innerHTML = compareList.map(sid => `
                <div class="compare-item">
                    <span>说话人 ${sid}</span>
                    <button class="remove-btn" onclick="toggleCompare(${sid})">移除</button>
                </div>
            `).join('');
        }
        
        function clearCompare() {
            compareList = [];
            updateComparePanel();
            renderSpeakers();
        }
        
        async function playCompare() {
            for (const sid of compareList) {
                const audio = document.getElementById(`audio-${sid}`);
                if (audio) {
                    audio.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    await audio.play();
                    await new Promise(r => { audio.onended = r; });
                }
            }
        }
        
        function exportList() {
            const data = {
                speakers: compareList,
                sentence: currentSentence,
                exported_at: new Date().toISOString()
            };
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `speakers_${new Date().getTime()}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }
        
        // 启动
        init();
    </script>
</body>
</html>"""

# ==================== 音频生成 ====================

proc writeWavFile(path: string, pcmData: seq[int16], sampleRate: int) =
  ## Write PCM data to WAV file
  let numChannels = 1.uint16
  let bitsPerSample = 16.uint16
  let byteRate = sampleRate.uint32 * numChannels.uint32 * (bitsPerSample div 8).uint32
  let blockAlign = numChannels * (bitsPerSample div 8)
  let dataSize = (pcmData.len * sizeof(int16)).uint32
  let chunkSize = 36.uint32 + dataSize

  var f = open(path, fmWrite)
  defer: f.close()

  f.write("RIFF")
  discard f.writeBuffer(chunkSize.addr, 4)
  f.write("WAVE")
  f.write("fmt ")
  let subchunkSize = 16.uint32
  discard f.writeBuffer(subchunkSize.addr, 4)
  let audioFormat = 1.uint16
  discard f.writeBuffer(audioFormat.addr, 2)
  discard f.writeBuffer(numChannels.addr, 2)
  let sr = sampleRate.uint32
  discard f.writeBuffer(sr.addr, 4)
  discard f.writeBuffer(byteRate.addr, 4)
  discard f.writeBuffer(blockAlign.addr, 2)
  discard f.writeBuffer(bitsPerSample.addr, 2)
  f.write("data")
  discard f.writeBuffer(dataSize.addr, 4)
  
  for sample in pcmData:
    discard f.writeBuffer(sample.addr, sizeof(int16))

proc generateSpeakerAudio(
  model: Model,
  lexicon: Table[string, LexiconEntry],
  tokens: Table[string, int64],
  sid: int,
  sentenceKey: string,
  sentenceText: string
): tuple[generated: bool, skipped: bool] =
  ## 为指定说话人生成指定句子的音频
  ## 返回: (是否新生成, 是否跳过)
  
  let filename = fmt"speaker_{sid:03d}_{sentenceKey}.wav"
  let filepath = OutputDir / filename
  
  # 检查文件是否已存在
  if fileExists(filepath):
    return (false, true)  # 跳过已存在的文件
  
  let tokenIds = textToTokenIds(sentenceText, lexicon, tokens)
  if tokenIds.len == 0:
    return (false, false)
  
  let output = runVitsTTS(
    model, tokenIds, sid = sid,
    noiseScale = 0.667'f32,
    lengthScale = 1.0'f32,
    noiseScaleW = 0.8'f32
  )
  
  if output.data.len == 0:
    return (false, false)
  
  let samples = output.toInt16Samples()
  writeWavFile(filepath, samples, SampleRate)
  return (true, false)

# ==================== 主程序 ====================

proc main() =
  echo "🎙️ VITS AISHELL3 说话人浏览器"
  echo "=============================="
  
  # 检查模型文件
  if not fileExists(ModelPath):
    echo "❌ 错误：模型文件不存在：", ModelPath
    quit(1)
  
  # 创建输出目录
  createDir(OutputDir)
  echo "📁 输出目录: ", OutputDir
  
  # 加载资源
  echo "\n📚 加载词典和词表..."
  let lexicon = loadLexicon(LexiconPath)
  let tokens = loadTokens(TokensPath)
  echo fmt"   词典: {lexicon.len} 条, 词表: {tokens.len} 个"
  
  # 加载模型
  echo "\n🤖 加载模型..."
  let model = loadModel(ModelPath)
  
  # 生成音频
  echo fmt"\n🎵 开始生成音频（{TotalSpeakers} 说话人 × {TestSentences.len} 句子）..."
  let startTime = epochTime()
  var totalGenerated = 0
  
  var totalSkipped = 0
  
  for sid in 0..<TotalSpeakers:
    let t0 = epochTime()
    var generated = 0
    var skipped = 0
    
    for (key, text) in TestSentences:
      let result = generateSpeakerAudio(model, lexicon, tokens, sid, key, text)
      if result.generated:
        inc(generated)
        inc(totalGenerated)
      elif result.skipped:
        inc(skipped)
        inc(totalSkipped)
    
    let dt = epochTime() - t0
    if sid mod 20 == 0 or sid == TotalSpeakers - 1:
      let pct = ((sid + 1).float / TotalSpeakers.float * 100).int
      let skipInfo = if skipped > 0: fmt" (跳过{skipped})" else: ""
      echo fmt"   [{pct:3d}%] 说话人 {sid:3d}: 生成{generated}, 共{TestSentences.len}{skipInfo} ({dt:.1f}s)"
  
  let totalTime = epochTime() - startTime
  echo fmt"\n✅ 完成！新生成 {totalGenerated} 个，跳过 {totalSkipped} 个已存在，耗时 {totalTime:.1f} 秒"
  
  # 生成 HTML
  echo "\n🌐 生成 HTML 页面..."
  
  # 准备 JS 数组
  var sentenceNames: seq[string]
  var sentenceTexts: seq[string]
  for (k, t) in TestSentences:
    sentenceNames.add("\"" & k & "\"")
    sentenceTexts.add("\"" & k & "\": \"" & t & "\"")
  
  var html = HtmlTemplate
  html = html.replace("${TOTAL_SPEAKERS}", $TotalSpeakers)
  html = html.replace("${MAX_SID}", $(TotalSpeakers - 1))
  html = html.replace("${TOTAL_SENTENCES}", $TestSentences.len)
  html = html.replace("${SENTENCES_JS_ARRAY}", "[" & sentenceNames.join(", ") & "]")
  html = html.replace("${SENTENCES_TEXT_JS}", "{" & sentenceTexts.join(", ") & "}")
  
  let htmlPath = OutputDir / "index.html"
  writeFile(htmlPath, html)
  echo "   页面已保存: ", htmlPath
  
  model.close()
  
  echo "\n🎉 全部完成！"
  echo "   请在浏览器中打开: ", htmlPath

when isMainModule:
  main()
