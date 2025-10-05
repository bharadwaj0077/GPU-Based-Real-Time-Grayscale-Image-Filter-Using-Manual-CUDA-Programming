


// ================================================================
// GPU-Based Grayscale Filter Server (CPU & GPU modes)
// ================================================================

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const app = express();

// ---------- PATH SETUP ----------
app.use(express.static(path.join(__dirname, 'public')));
app.use('/results', express.static(path.join(__dirname, 'results')));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// ---------- STORAGE CONFIG ----------
const upload = multer({
  dest: path.join(__dirname, 'uploads')
});

// ---------- EXECUTABLE MAP ----------
const exeMap = {
  image: { cpu: 'cpu_gray_image.exe', gpu: 'cuda_gray_image.exe' },
  video: { cpu: 'cpu_gray_video.exe', gpu: 'cuda_gray_video.exe' }
};

// ---------- PROCESS ENDPOINT ----------
app.post('/api/process', upload.single('file'), (req, res) => {
  try {
    const { engine, media } = req.query;
    const file = req.file;

    if (!file) {
      return res.status(400).json({ ok: false, error: 'No file uploaded' });
    }

    const exe = exeMap[media]?.[engine];
    if (!exe) {
      return res.status(400).json({ ok: false, error: 'Invalid engine or media type' });
    }

    const inPath = file.path;
    const outName = `${Date.now()}_${path.basename(file.originalname, path.extname(file.originalname))}_gray${media === 'video' ? '.mp4' : '.jpg'}`;
    const outPath = path.join(__dirname, 'results', outName);

    // Spawn the process
    console.log(`🎬 Running ${exe} (${engine.toUpperCase()}) on ${media}:`);
    console.log(`Input: ${inPath}`);
    console.log(`Output: ${outPath}`);

    const processObj = spawn(path.join(__dirname, 'bin', exe), [inPath, outPath]);

    let stdout = '';
    let stderr = '';

    processObj.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    processObj.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    processObj.on('close', (code) => {
      console.log(`🎬 Process closed with code: ${code}`);
      const exists = fs.existsSync(outPath);

      // Extract time
      let timeMs = null;
      const match = stdout.match(/TIME_TAKEN_MS=(\d+(\.\d+)?)/);
      if (match) timeMs = parseFloat(match[1]);

      if (code === 0 && exists) {
        return res.json({
          ok: true,
          outUrl: `/results/${path.basename(outPath)}`,
          timeMs,
          engine,
          media
        });
      } else {
        return res.status(500).json({
          ok: false,
          error: `Processing failed (code: ${code})`,
          stderr,
          stdout
        });
      }
    });
  } catch (err) {
    console.error('❌ Server error:', err);
    res.status(500).json({ ok: false, error: 'Internal server error' });
  }
});

// ---------- HOME ROUTE ----------
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'main.html'));
});

// ---------- SERVER START ----------
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`✅ Running on http://localhost:${PORT}`);
});
