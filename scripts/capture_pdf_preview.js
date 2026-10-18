const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

async function main() {
    const artifactDir = "C:\\Users\\harsh\\.gemini\\antigravity\\brain\\bbcd98a0-900e-4b53-8e4b-456b213399b5";
    
    // We can run python to get the HTML or use a dumped HTML file
    const browser = await chromium.launch({
        channel: 'chrome',
        headless: true
    });

    const page = await browser.newPage({
        viewport: { width: 1200, height: 1600 },
        deviceScaleFactor: 2
    });

    // Check if test_html exists or generate it
    let htmlPath = path.resolve("test_report.html");
    if (!fs.existsSync(htmlPath)) {
        console.error("test_report.html does not exist");
        process.exit(1);
    }

    const htmlContent = fs.readFileSync(htmlPath, 'utf-8');
    await page.setContent(htmlContent, { waitUntil: 'load' });
    await page.waitForTimeout(600);

    const pages = page.locator(".pdf-page");
    const count = await pages.count();
    console.log(`Found ${count} pages in HTML`);

    for (let i = 0; i < count; i++) {
        const outPath = path.join(artifactDir, `pdf_preview_page${i+1}.png`);
        await pages.nth(i).screenshot({ path: outPath });
        console.log(`Saved page ${i+1} preview to ${outPath}`);
    }

    await browser.close();
}

main().catch(err => {
    console.error(err);
    process.exit(1);
});
