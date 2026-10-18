const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

async function runFakeVerification() {
    const artifactDir = "C:\\Users\\harsh\\.gemini\\antigravity\\brain\\bbcd98a0-900e-4b53-8e4b-456b213399b5";
    const testImage = path.resolve('test_images/fake/test_fake_c01276_c01230.jpg');
    
    console.log(`Starting Live Fake Image Verification on Vercel with: ${testImage}`);
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
    await page.waitForTimeout(600);

    console.log("Clicking #run-image-btn to launch scan...");
    await page.click('#run-image-btn');

    console.log("Waiting for window.lastAnalysisResult...");
    await page.waitForFunction(() => window.lastAnalysisResult !== null && window.lastAnalysisResult !== undefined, { timeout: 15000 });
    console.log("Client forensics analysis completed for fake image!");

    await page.waitForTimeout(1000);
    await page.screenshot({ path: path.join(artifactDir, 'live_fake_01_alignment.png'), fullPage: false });

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
    await page.screenshot({ path: path.join(artifactDir, 'live_fake_02_results_card.png'), fullPage: false });

    const reportHtml = await page.evaluate(() => {
        if (window.generateClientReportHtml && window.lastAnalysisResult) {
            return window.generateClientReportHtml(window.lastAnalysisResult);
        }
        return null;
    });

    if (reportHtml) {
        console.log("Generated dynamic 4-page report HTML from uploaded fake image!");
        const reportPath = path.join(artifactDir, 'live_fake_report.html');
        fs.writeFileSync(reportPath, reportHtml, 'utf8');

        const reportPage = await context.newPage();
        await reportPage.setContent(reportHtml, { waitUntil: 'networkidle' });
        await reportPage.screenshot({ path: path.join(artifactDir, 'live_fake_dossier_page1.png'), fullPage: false });
        console.log("Saved fake dossier preview screenshot!");
    }

    console.log("Fake verification completed successfully!");
    await browser.close();
}

runFakeVerification().catch(err => {
    console.error("Verification failed:", err);
    process.exit(1);
});
