PDF Scanner / Invoice Renamer

This small utility renames and sorts scanned invoice PDFs into an output folder structure by business and vendor.

Filename format produced:
  <Scan Date> <Business Name> <Vendor name> <Amount>.pdf

Where Scan Date is taken from the PDF file modification time when possible (YYYYMMDD).

Quick start (headless):
1. Edit or create `config.json` in the project root. Example:
{
  "openai_api_key": "sk-...",
  "watch_folder": "/path/to/incoming",
  "output_folder": "/path/to/sorted",
  "business_name": "My Business"
}

2. Install dependencies:

   python -m pip install -r requirements.txt

3. Run in headless mode to process all PDFs in the watch folder:

   python main.py --nogui

GUI mode:

   python main.py

Notes:
- The application uses OpenAI to analyze invoice text. Set `openai_api_key` in `config.json` or configure via the GUI.
- Files are moved into `output_folder/<Business_Name>/<Vendor_Name>/` with safe filenames.
- Duplicate filenames are handled by appending a counter.

If your PDFs are scanned images without embedded text, OCR is required (not included by default).