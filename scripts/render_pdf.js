const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

async function main() {
    const args = process.argv.slice(2);
    if (args.length < 2) {
        console.error("Usage: node render_pdf.js <input_html_or_json> <output_pdf>");
        process.exit(1);
    }

    const inputPath = path.resolve(args[0]);
    const outputPath = path.resolve(args[1]);

    let htmlContent = "";
    if (inputPath.endsWith(".html")) {
        htmlContent = fs.readFileSync(inputPath, 'utf-8');
    } else {
        console.error("Input must be an HTML file.");
        process.exit(1);
    }

    const browser = await chromium.launch({
        channel: 'chrome',
        headless: true
    });

    try {
        const page = await browser.newPage();
        await page.setContent(htmlContent, { waitUntil: 'load' });
        await page.waitForTimeout(400); // Allow fonts and images to settle

        await page.pdf({
            path: outputPath,
            format: 'A4',
            printBackground: true,
            margin: {
                top: '0mm',
                right: '0mm',
                bottom: '0mm',
                left: '0mm'
            },
            preferCSSPageSize: true
        });

        console.log(`[SUCCESS] PDF saved to ${outputPath}`);
    } finally {
        await browser.close();
    }
}

main().catch(err => {
    console.error("[ERROR]", err);
    process.exit(1);
});
