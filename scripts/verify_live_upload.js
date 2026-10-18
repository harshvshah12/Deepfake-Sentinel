const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

async function runVerification() {
    const artifactDir = "C:\\Users\\harsh\\.gemini\\antigravity\\brain\\bbcd98a0-900e-4b53-8e4b-456b213399b5";
    const testImage = path.resolve('test_images/real/test_real_biden.jpg');
    
    console.log(`Starting Live Verification on Vercel with test image: ${testImage}`);
    const browser = await chromium.launch({ headless: true, channel: 'msedge' });
    const context = await browser.newContext({ viewport: { width: 1440, height: 960 } });
    const page = await context.newPage();

    page.on('console', msg => {
        console.log(`[PAGE LOG] ${msg.type()}: ${msg.text()}`);
    });

    console.log("Navigating to https://deepfakesentinel-kappa.vercel.app...");
    await page.goto('https://deepfakesentinel-kappa.vercel.app', { waitUntil: 'networkidle' });

    console.log("Setting input files on #image-file-input...");
    await page.setInputFiles('#image-file-input', testImage);

    // Wait 500ms for preview to render
    await page.waitForTimeout(600);

    console.log("Clicking #run-image-btn to launch scan...");
    await page.click('#run-image-btn');

    // Wait for client forensics analysis to complete and lastAnalysisResult to be populated
    console.log("Waiting for window.lastAnalysisResult...");
    await page.waitForFunction(() => window.lastAnalysisResult !== null && window.lastAnalysisResult !== undefined, { timeout: 15000 });
    console.log("Client forensics analysis completed! Proceeding to capture chapters...");

    // Wait for theatre to appear
    await page.waitForSelector('#scan-cinema:not(.hidden)', { timeout: 10000 });

    // Take screenshot of Step 01 Ingest / Wait state
    await page.waitForTimeout(1000);
    await page.screenshot({ path: path.join(artifactDir, 'live_biden_01_ingest.png'), fullPage: false });

    // Jump to Alignment step (Step 6 / index 5)
    await page.evaluate(() => {
        if (typeof cinemaJumpTo === 'function') cinemaJumpTo(5);
    });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: path.join(artifactDir, 'live_biden_02_alignment.png'), fullPage: false });

    // Switch alignment mode to split slider
    await page.evaluate(() => {
        if (typeof setAlignMode === 'function') setAlignMode('split');
    });
    await page.waitForTimeout(600);
    await page.screenshot({ path: path.join(artifactDir, 'live_biden_03_alignment_split.png'), fullPage: false });

    // Jump to 2D FFT step (index 9)
    await page.evaluate(() => {
        if (typeof cinemaJumpTo === 'function') cinemaJumpTo(9);
    });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: path.join(artifactDir, 'live_biden_04_fft_spectrum.png'), fullPage: false });

    // Jump to Grad-CAM step (index 10)
    await page.evaluate(() => {
        if (typeof cinemaJumpTo === 'function') cinemaJumpTo(10);
    });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: path.join(artifactDir, 'live_biden_05_gradcam.png'), fullPage: false });

    // Jump to Iris Zoom step (index 11)
    await page.evaluate(() => {
        if (typeof cinemaJumpTo === 'function') cinemaJumpTo(11);
    });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: path.join(artifactDir, 'live_biden_06_iris_zoom.png'), fullPage: false });

    // Jump to Final Consensus Decision (index 12)
    await page.evaluate(() => {
        if (typeof cinemaJumpTo === 'function') cinemaJumpTo(12);
    });
    await page.waitForTimeout(1200);
    await page.screenshot({ path: path.join(artifactDir, 'live_biden_07_consensus.png'), fullPage: false });

    // Scroll to results card
    await page.evaluate(() => {
        if (typeof renderImageResults === 'function' && window.lastAnalysisResult) {
            renderImageResults(window.lastAnalysisResult);
        }
        const card = document.getElementById('image-result-card');
        if (card) {
            card.classList.remove('hidden');
            card.scrollIntoView();
        }
    });
    await page.waitForTimeout(800);
    await page.screenshot({ path: path.join(artifactDir, 'live_biden_08_results_card.png'), fullPage: false });

    // Render client-side 4-page report and take full screenshot
    const reportHtml = await page.evaluate(() => {
        if (window.generateClientReportHtml && window.lastAnalysisResult) {
            return window.generateClientReportHtml(window.lastAnalysisResult);
        }
        return null;
    });

    if (reportHtml) {
        console.log("Generated dynamic 4-page report HTML from uploaded image!");
        const reportPath = path.join(artifactDir, 'live_biden_report.html');
        fs.writeFileSync(reportPath, reportHtml, 'utf8');

        // Open report in new page to screenshot all pages
        const reportPage = await context.newPage();
        await reportPage.setContent(reportHtml, { waitUntil: 'networkidle' });
        await reportPage.screenshot({ path: path.join(artifactDir, 'live_biden_dossier_page1.png'), fullPage: false });
        console.log("Saved dossier preview screenshot!");
    }

    console.log("Verification completed successfully!");
    await browser.close();
}

runVerification().catch(err => {
    console.error("Verification failed:", err);
    process.exit(1);
});
