const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

async function captureDemo() {
    const artifactDir = "C:\\Users\\harsh\\.gemini\\antigravity\\brain\\bbcd98a0-900e-4b53-8e4b-456b213399b5";
    if (!fs.existsSync(artifactDir)) {
        fs.mkdirSync(artifactDir, { recursive: true });
    }

    console.log("Launching browser...");
    const browser = await chromium.launch({
        channel: 'chrome',
        headless: true
    });
    const context = await browser.newContext({
        viewport: { width: 1440, height: 1050 },
        deviceScaleFactor: 2 // High resolution Retina quality
    });
    const page = await context.newPage();

    console.log("Navigating to http://127.0.0.1:8000 ...");
    await page.goto("http://127.0.0.1:8000", { waitUntil: "networkidle" });

    const sampleImg = path.resolve("dataset", "real", "real_0.jpg");
    console.log(`Uploading test specimen: ${sampleImg}`);
    await page.setInputFiles("#image-file-input", sampleImg);

    const predictPromise = page.waitForResponse(
        res => res.url().includes("/predict") && res.status() === 200,
        { timeout: 90000 }
    );

    console.log("Triggering Forensic Scan...");
    await page.click("#run-image-btn");
    await predictPromise;
    console.log("Inference complete! Waiting for scan cinema to populate...");

    await page.waitForSelector("#scan-cinema:not(.hidden)", { timeout: 15000 });
    await page.waitForTimeout(600);

    // Pause autoplay immediately so we can inspect and capture each step
    await page.evaluate(() => {
        if (typeof cinemaState !== 'undefined' && cinemaState.isPlaying) {
            toggleCinemaPlay();
        }
    });

    const story = await page.evaluate(() => {
        return cinemaState.story.map((s, i) => ({ index: i, id: s.id, kicker: s.kicker, title: s.title }));
    });
    console.log("Captured story chapters:", story);

    const findIndex = (id) => story.findIndex(s => s.id === id);
    const cinemaElement = page.locator("#scan-cinema");

    // 1. Alignment Dual Side-by-Side
    const alignIdx = findIndex("align");
    if (alignIdx !== -1) {
        console.log(`Capturing Alignment step (index ${alignIdx}) - Dual View...`);
        await page.evaluate((idx) => cinemaJumpTo(idx), alignIdx);
        await page.waitForTimeout(1000);
        await cinemaElement.screenshot({
            path: path.join(artifactDir, "demo_01_alignment_dual.png")
        });

        // Toggle to Split slider mode
        console.log("Switching to split slider mode...");
        await page.evaluate(() => {
            if (typeof setAlignMode === 'function') {
                setAlignMode('slider');
            }
        });
        await page.waitForTimeout(600);
        await cinemaElement.screenshot({
            path: path.join(artifactDir, "demo_02_alignment_split_slider.png")
        });
    }

    // 2. Tensor Preprocessing
    const tensorIdx = findIndex("tensor");
    if (tensorIdx !== -1) {
        console.log(`Capturing Tensor step (index ${tensorIdx})...`);
        await page.evaluate((idx) => cinemaJumpTo(idx), tensorIdx);
        await page.waitForTimeout(1000);
        await cinemaElement.screenshot({
            path: path.join(artifactDir, "demo_03_tensor_normalization.png")
        });
    }

    // 3. ViT 196-Patch Grid
    const vitIdx = findIndex("vit_patches");
    if (vitIdx !== -1) {
        console.log(`Capturing ViT Patches step (index ${vitIdx})...`);
        await page.evaluate((idx) => cinemaJumpTo(idx), vitIdx);
        await page.waitForTimeout(1200);
        await cinemaElement.screenshot({
            path: path.join(artifactDir, "demo_04_vit_patches_matrix.png")
        });
    }

    // 4. 2D FFT Optical Spectrum
    const fftIdx = findIndex("fft");
    if (fftIdx !== -1) {
        console.log(`Capturing FFT step (index ${fftIdx})...`);
        await page.evaluate((idx) => cinemaJumpTo(idx), fftIdx);
        await page.waitForTimeout(1000);
        await cinemaElement.screenshot({
            path: path.join(artifactDir, "demo_05_fft_spectrum.png")
        });
    }

    // 5. Grad-CAM Thermal Activation
    const camIdx = findIndex("cam");
    if (camIdx !== -1) {
        console.log(`Capturing Grad-CAM step (index ${camIdx})...`);
        await page.evaluate((idx) => cinemaJumpTo(idx), camIdx);
        await page.waitForTimeout(1000);
        await cinemaElement.screenshot({
            path: path.join(artifactDir, "demo_06_gradcam_overlay.png")
        });
    }

    // 6. Decision & Verdict
    const decIdx = findIndex("decision");
    if (decIdx !== -1) {
        console.log(`Capturing Decision step (index ${decIdx})...`);
        await page.evaluate((idx) => cinemaJumpTo(idx), decIdx);
        await page.waitForTimeout(1000);
        await cinemaElement.screenshot({
            path: path.join(artifactDir, "demo_07_consensus_decision.png")
        });
    }

    // 7. Full page overview of the interactive theatre
    console.log("Capturing full page overview...");
    await page.screenshot({
        path: path.join(artifactDir, "demo_08_full_theatre_dashboard.png"),
        fullPage: false
    });

    console.log("All screenshots successfully captured!");
    await browser.close();
}

captureDemo().catch(err => {
    console.error("Error capturing demo:", err);
    process.exit(1);
});
