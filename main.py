import streamlit as st

# MUST be first Streamlit command
st.set_page_config(page_title="Invoice Data Extractor", page_icon="📊", layout="wide")

import os
import json
from datetime import datetime, timedelta
import logging
import tempfile
import zipfile
import io
import pandas as pd
import re
import base64

# PDF processing
try:
    from pypdf import PdfReader

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Anthropic Claude API
try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Image conversion for vision API
try:
    from pdf2image import convert_from_path

    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class InvoiceProcessor:
    def __init__(self):
        self.client = None
        self.model = "claude-sonnet-4-20250514"  # Best for vision/OCR tasks
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_client(self, api_key):
        """Initialize Anthropic client with API key"""
        if not ANTHROPIC_AVAILABLE:
            return False, "❌ Anthropic library not installed. Run: pip install anthropic"

        if api_key and api_key.startswith('sk-ant-'):
            try:
                self.client = anthropic.Anthropic(api_key=api_key)

                # Test the API key with a simple request
                test_response = self.client.messages.create(
                    model=self.model,
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hi"}]
                )
                return True, f"✅ Claude API initialized ({self.model})"

            except anthropic.AuthenticationError:
                return False, "❌ Invalid API key"
            except anthropic.RateLimitError:
                return False, "❌ Rate limit exceeded"
            except Exception as e:
                return False, f"❌ API error: {str(e)[:100]}"
        else:
            return False, "⚠️ Please enter a valid Anthropic API key (should start with 'sk-ant-')"

    def pdf_to_base64_image(self, pdf_path, max_size_bytes=4_500_000):
        """Convert first page of PDF to base64 image for vision API, ensuring under 5MB limit"""
        if not PDF2IMAGE_AVAILABLE or not PIL_AVAILABLE:
            return None, None

        try:
            # Start with high DPI for quality
            dpi = 250
            images = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)

            if not images:
                return None, None

            img = images[0]

            # Try JPEG first (much smaller than PNG)
            for quality in [85, 70, 55, 40]:
                buffered = io.BytesIO()

                # Convert to RGB if necessary (JPEG doesn't support transparency)
                if img.mode in ('RGBA', 'LA', 'P'):
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = rgb_img

                img.save(buffered, format="JPEG", quality=quality, optimize=True)
                img_bytes = buffered.getvalue()

                if len(img_bytes) <= max_size_bytes:
                    self.logger.info(f"Image size: {len(img_bytes) / 1024 / 1024:.2f}MB (quality={quality})")
                    img_base64 = base64.standard_b64encode(img_bytes).decode('utf-8')
                    return img_base64, "image/jpeg"

            # If still too large, resize the image
            for scale in [0.75, 0.6, 0.5, 0.4]:
                new_size = (int(img.width * scale), int(img.height * scale))
                resized_img = img.resize(new_size, Image.LANCZOS)

                buffered = io.BytesIO()
                resized_img.save(buffered, format="JPEG", quality=70, optimize=True)
                img_bytes = buffered.getvalue()

                if len(img_bytes) <= max_size_bytes:
                    self.logger.info(f"Image size after resize: {len(img_bytes) / 1024 / 1024:.2f}MB (scale={scale})")
                    img_base64 = base64.standard_b64encode(img_bytes).decode('utf-8')
                    return img_base64, "image/jpeg"

            # Last resort: very aggressive compression
            self.logger.warning("Using aggressive compression for large image")
            new_size = (int(img.width * 0.3), int(img.height * 0.3))
            resized_img = img.resize(new_size, Image.LANCZOS)
            buffered = io.BytesIO()
            resized_img.save(buffered, format="JPEG", quality=50, optimize=True)
            img_bytes = buffered.getvalue()
            img_base64 = base64.standard_b64encode(img_bytes).decode('utf-8')
            return img_base64, "image/jpeg"

        except Exception as e:
            self.logger.error(f"Error converting PDF to image: {e}")

        return None, None

    def extract_invoice_data_with_claude(self, pdf_path, filename):
        """Use Claude Vision to extract invoice data from PDF image"""
        if not self.client:
            return None

        img_base64, media_type = self.pdf_to_base64_image(pdf_path)
        if not img_base64:
            self.logger.warning("Could not convert PDF to image")
            return None

        prompt = """You are a precise document data extractor. Carefully examine this document image and extract the data fields listed below.

RETURN ONLY THIS EXACT JSON FORMAT (use null for any field you cannot clearly read or are unsure about):
{
    "vendor_name": "string or null",
    "business_name": "string or null",
    "invoice_number": "string or null",
    "amount": number or null,
    "invoice_date": "YYYY-MM-DD or null",
    "payment_terms": "string or null",
    "due_date": "YYYY-MM-DD or null",
    "notes": "string or null"
}

FIELD DEFINITIONS AND WHERE TO FIND THEM:

1. VENDOR_NAME: The organization that ISSUED/SENT this document
   - Location: TOP of document, letterhead, logo area, return address
   - This is who is billing or sending the notice
   - Examples: "The Municipal Authority of Township of Westfall", "Pennsylvania Department of Revenue", "Erie Insurance Company"

2. BUSINESS_NAME: The organization/person RECEIVING this document
   - Location: "Bill To:", "To:", "Customer:", "Ship To:" sections
   - This is who owes money or is being notified
   - Usually in a box or indented section below the header

3. INVOICE_NUMBER: The document's unique reference number
   - Location: Usually upper right, near the date
   - Labels to look for: "Invoice #", "Invoice Number", "Inv #", "Letter ID", "Document #", "Check No.", "Reference #"
   - Extract the actual number/code

4. AMOUNT: The TOTAL amount - this is critical, look carefully
   - Location: BOTTOM of document, usually right-aligned
   - Labels to look for: "Total", "Amount Due", "Balance Due", "Total Due", "Grand Total", "Payment Amount"
   - Extract as a number WITHOUT currency symbols or commas (e.g., 520.91 not $520.91)
   - If multiple amounts exist, use the FINAL TOTAL at the very bottom

5. INVOICE_DATE: The date this document was created/issued
   - Location: Upper right area, near invoice number
   - Labels: "Date", "Invoice Date", "Date Issued", "Document Date"
   - Convert to YYYY-MM-DD format

6. PAYMENT_TERMS: How long the recipient has to pay
   - Location: Often in a row with other details like P.O. Number
   - Labels: "Terms", "Payment Terms", "Net"
   - Examples: "10 days", "Net 30", "Due on Receipt", "Net 15"

7. DUE_DATE: When payment is actually due
   - If explicitly shown: Look for "Due Date", "Payment Due", "Due By"
   - If NOT explicitly shown but you have invoice_date and payment_terms: CALCULATE IT
     - Example: invoice_date "2025-05-31" + terms "10 days" = due_date "2025-06-10"
   - Convert to YYYY-MM-DD format

8. NOTES: Any special status indicators, stamps, or warnings
   - Look for: "PAST DUE", "PAID", "FINAL NOTICE", "OVERDUE", "URGENT", "NON-NEGOTIABLE", "VOID"
   - Also note if this is not a standard invoice: "Check", "Refund", "Credit Memo", "Notice"
   - If nothing special, use null

CRITICAL INSTRUCTIONS:
- Read the document VERY CAREFULLY - accuracy is essential
- The TOTAL/AMOUNT is usually at the BOTTOM of the document in a prominent position
- Use null for any field you cannot clearly see - do NOT guess
- Dates must be in YYYY-MM-DD format
- Amounts must be numbers only (no $ or commas)
- Return ONLY the JSON object, no other text"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=800,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": img_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            result = response.content[0].text.strip()

            # Remove markdown code blocks if present
            result = re.sub(r'```json\s*|\s*```', '', result).strip()

            self.logger.info(f"Claude response for {filename}: {result[:300]}")

            data = json.loads(result)
            return self._validate_and_clean_data(data)

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse error for {filename}: {e}")
            self.logger.error(f"Raw response: {result[:500]}")
            return None
        except Exception as e:
            self.logger.error(f"Claude API error: {e}")
            return None

    def _validate_and_clean_data(self, data):
        """Validate and clean extracted data"""

        # Text fields - keep as None if not found
        for field in ['vendor_name', 'business_name', 'invoice_number', 'payment_terms', 'notes']:
            if not data.get(field) or data.get(field) in ['null', 'None', '']:
                data[field] = None

        # Handle amount
        if data.get('amount') is not None and data.get('amount') not in ['null', 'None', '']:
            try:
                amount_str = str(data['amount']).replace('$', '').replace(',', '').strip()
                data['amount'] = float(amount_str)
            except:
                data['amount'] = None
        else:
            data['amount'] = None

        # Validate date formats
        for date_field in ['invoice_date', 'due_date']:
            if data.get(date_field) and data.get(date_field) not in ['null', 'None', '']:
                try:
                    parsed_date = pd.to_datetime(data[date_field])
                    data[date_field] = parsed_date.strftime('%Y-%m-%d')
                except:
                    data[date_field] = None
            else:
                data[date_field] = None

        # Calculate due_date if we have invoice_date and payment_terms but no due_date
        if data.get('invoice_date') and data.get('payment_terms') and not data.get('due_date'):
            data['due_date'] = self._calculate_due_date(data['invoice_date'], data['payment_terms'])

        return data

    def _calculate_due_date(self, invoice_date_str, terms):
        """Calculate due date from invoice date and payment terms"""
        try:
            invoice_date = datetime.strptime(invoice_date_str, '%Y-%m-%d')
            terms_lower = str(terms).lower()

            # Match patterns like "10 days", "net 30", etc.
            match = re.search(r'(\d+)\s*(?:days?|day)', terms_lower)
            if match:
                days = int(match.group(1))
                due_date = invoice_date + timedelta(days=days)
                return due_date.strftime('%Y-%m-%d')

            # Match "net XX" pattern
            match = re.search(r'net\s*(\d+)', terms_lower)
            if match:
                days = int(match.group(1))
                due_date = invoice_date + timedelta(days=days)
                return due_date.strftime('%Y-%m-%d')

        except Exception as e:
            self.logger.error(f"Error calculating due date: {e}")

        return None

    def process_single_pdf(self, file_content, filename):
        """Process a single PDF file and extract invoice data"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name

            invoice_data = None

            if PDF2IMAGE_AVAILABLE:
                self.logger.info(f"Processing {filename} with Claude Vision")
                invoice_data = self.extract_invoice_data_with_claude(tmp_path, filename)

            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

            if not invoice_data:
                return {
                    'filename': filename,
                    'vendor_name': None,
                    'business_name': None,
                    'invoice_number': None,
                    'amount': None,
                    'invoice_date': None,
                    'payment_terms': None,
                    'due_date': None,
                    'notes': None,
                    'date_processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'Failed - Could not extract data'
                }

            invoice_data['filename'] = filename
            invoice_data['date_processed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            invoice_data['status'] = 'Success'

            return invoice_data

        except Exception as e:
            self.logger.error(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'filename': filename,
                'vendor_name': None,
                'business_name': None,
                'invoice_number': None,
                'amount': None,
                'invoice_date': None,
                'payment_terms': None,
                'due_date': None,
                'notes': None,
                'date_processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': f'Failed - {str(e)[:100]}'
            }


def main():
    if not PDF_AVAILABLE:
        st.error("❌ pypdf is not installed. Run: `pip install pypdf`")
        st.stop()

    if not ANTHROPIC_AVAILABLE:
        st.error("❌ anthropic is not installed. Run: `pip install anthropic`")
        st.stop()

    if not PDF2IMAGE_AVAILABLE:
        st.error("❌ pdf2image is not installed. Run: `pip install pdf2image`")
        st.info("Also install poppler: `apt-get install poppler-utils` (Linux) or `brew install poppler` (Mac)")
        st.stop()

    st.title("📊 Invoice Data Extractor")
    st.markdown("### Powered by Claude Vision AI")

    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = InvoiceProcessor()
    if 'api_key_valid' not in st.session_state:
        st.session_state.api_key_valid = False
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

    # Sidebar
    with st.sidebar:
        st.header("🔑 API Configuration")

        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            help="Get your API key from console.anthropic.com"
        )

        if st.button("Validate API Key", use_container_width=True):
            if api_key:
                with st.spinner("Validating..."):
                    success, message = st.session_state.processor.initialize_client(api_key)
                    if success:
                        st.session_state.api_key_valid = True
                        st.session_state.api_key = api_key
                        st.success(message)
                    else:
                        st.session_state.api_key_valid = False
                        st.error(message)
            else:
                st.error("Please enter an API key")

        if st.session_state.get('api_key_valid', False):
            st.success("✅ Connected to Claude")
        else:
            st.warning("⚠️ API Key Required")

        st.markdown("---")

        st.header("ℹ️ About")
        st.markdown(f"""
        **Model:** Claude Sonnet 4

        **Extracted Fields:**
        - Vendor Name
        - Business Name
        - Invoice Number
        - Amount (Total)
        - Invoice Date
        - Payment Terms
        - Due Date (calculated)
        - Notes/Status
        """)

        st.markdown("---")
        st.markdown("""
        **Tips for best results:**
        - Use clear, high-quality scans
        - Ensure document is right-side up
        - Single page invoices work best
        """)

    # Main content
    if not st.session_state.get('api_key_valid', False):
        st.warning("⚠️ Please enter your Anthropic API key in the sidebar to continue.")

        st.markdown("""
        ### Getting Started

        1. **Get an API Key:** Visit [console.anthropic.com](https://console.anthropic.com) to create an account and get your API key
        2. **Enter the Key:** Paste your API key in the sidebar (starts with `sk-ant-`)
        3. **Upload Invoices:** Upload your PDF invoices individually or as a ZIP file
        4. **Download Results:** Get your extracted data as a CSV file

        ### Why Claude Vision?

        Claude's vision capabilities excel at:
        - Reading scanned documents with high accuracy
        - Understanding document layouts and structure
        - Extracting data from tables and forms
        - Handling varied invoice formats
        """)
        st.stop()

    # Show results if processing is complete
    if st.session_state.results_df is not None and st.session_state.processing_complete:
        st.header("✅ Processing Complete!")

        df = st.session_state.results_df

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", len(df))
        with col2:
            successful = len(df[df['status'] == 'Success'])
            st.metric("Successful", successful)
        with col3:
            total_amount = df['amount'].sum() if df['amount'].notna().any() else 0
            st.metric("Total Amount", f"${total_amount:,.2f}")
        with col4:
            past_due_count = 0
            if 'notes' in df.columns:
                past_due_count = len(df[df['notes'].str.contains('PAST DUE', case=False, na=False)])
            st.metric("Past Due", past_due_count)

        # Data table
        st.subheader("📋 Extracted Data")

        display_cols = [
            'filename', 'vendor_name', 'business_name', 'invoice_number',
            'amount', 'invoice_date', 'payment_terms', 'due_date', 'notes', 'status'
        ]
        display_df = df[[c for c in display_cols if c in df.columns]]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "filename": st.column_config.TextColumn("File", width="medium"),
                "vendor_name": st.column_config.TextColumn("Vendor", width="medium"),
                "business_name": st.column_config.TextColumn("Business", width="medium"),
                "invoice_number": st.column_config.TextColumn("Invoice #", width="small"),
                "amount": st.column_config.NumberColumn("Amount", format="$%.2f", width="small"),
                "invoice_date": st.column_config.DateColumn("Invoice Date", format="YYYY-MM-DD", width="small"),
                "payment_terms": st.column_config.TextColumn("Terms", width="small"),
                "due_date": st.column_config.DateColumn("Due Date", format="YYYY-MM-DD", width="small"),
                "notes": st.column_config.TextColumn("Notes", width="small"),
                "status": st.column_config.TextColumn("Status", width="small"),
            }
        )

        # Action buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"invoice_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

        with col2:
            # Excel download
            try:
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_buffer.seek(0)
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_buffer,
                    file_name=f"invoice_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except ImportError:
                st.button("📥 Excel (install openpyxl)", disabled=True, use_container_width=True)

        with col3:
            if st.button("🔄 Process New Files", use_container_width=True):
                st.session_state.results_df = None
                st.session_state.processing_complete = False
                st.rerun()

        # Additional analysis
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 By Vendor")
            if df['vendor_name'].notna().any():
                vendor_summary = df[df['vendor_name'].notna()].groupby('vendor_name').agg({
                    'amount': 'sum',
                    'filename': 'count'
                }).rename(columns={'filename': 'invoices'}).sort_values('amount', ascending=False)
                st.dataframe(
                    vendor_summary,
                    column_config={
                        "amount": st.column_config.NumberColumn("Total", format="$%.2f"),
                        "invoices": st.column_config.NumberColumn("Count")
                    }
                )
            else:
                st.info("No vendor data available")

        with col2:
            st.subheader("⚠️ Needs Attention")
            attention_items = df[
                (df['notes'].notna() & (df['notes'] != '')) |
                (df['amount'].isna()) |
                (df['status'] != 'Success')
                ]
            if len(attention_items) > 0:
                st.dataframe(
                    attention_items[['filename', 'notes', 'amount', 'status']],
                    hide_index=True,
                    column_config={
                        "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                    }
                )
            else:
                st.success("✅ All items processed successfully!")

    else:
        # Upload section
        st.header("📤 Upload Invoice Files")

        upload_type = st.radio(
            "Upload method:",
            ["Individual PDF files", "ZIP file (folder of PDFs)"],
            horizontal=True
        )

        files_to_process = []

        if upload_type == "Individual PDF files":
            uploaded_files = st.file_uploader(
                "Upload PDF invoices",
                type="pdf",
                accept_multiple_files=True,
                help="Select one or more PDF invoice files"
            )
            if uploaded_files:
                for f in uploaded_files:
                    files_to_process.append({
                        'name': f.name,
                        'content': f.getvalue()
                    })
        else:
            zip_file = st.file_uploader(
                "Upload ZIP file containing PDFs",
                type="zip",
                help="Upload a ZIP file containing your invoice PDFs"
            )
            if zip_file:
                try:
                    with zipfile.ZipFile(io.BytesIO(zip_file.read())) as z:
                        pdf_files = [f for f in z.namelist() if
                                     f.lower().endswith('.pdf') and not f.startswith('__MACOSX')]
                        st.success(f"Found {len(pdf_files)} PDF files in ZIP")
                        for pdf_name in pdf_files:
                            files_to_process.append({
                                'name': os.path.basename(pdf_name),
                                'content': z.read(pdf_name)
                            })
                except Exception as e:
                    st.error(f"Error reading ZIP: {e}")

        if files_to_process:
            st.success(f"📄 Ready to process {len(files_to_process)} file(s)")

            with st.expander("📁 View files to process"):
                for i, file in enumerate(files_to_process):
                    st.write(f"{i + 1}. {file['name']}")

            if st.button("🚀 Process Invoices", type="primary", use_container_width=True):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, file in enumerate(files_to_process):
                    progress = (i + 1) / len(files_to_process)
                    progress_bar.progress(progress)
                    status_text.text(f"🔍 Processing {i + 1}/{len(files_to_process)}: {file['name']}")

                    result = st.session_state.processor.process_single_pdf(file['content'], file['name'])
                    results.append(result)

                # Create DataFrame
                df = pd.DataFrame(results)

                # Reorder columns
                column_order = [
                    'filename', 'vendor_name', 'business_name', 'invoice_number',
                    'amount', 'invoice_date', 'payment_terms', 'due_date', 'notes',
                    'date_processed', 'status'
                ]
                df = df[[c for c in column_order if c in df.columns]]

                st.session_state.results_df = df
                st.session_state.processing_complete = True

                progress_bar.empty()
                status_text.empty()

                st.success(f"✅ Processed {len(results)} invoices!")
                st.balloons()
                st.rerun()

        else:
            # Instructions when no files uploaded
            st.markdown("""
            ### 📋 How to Use

            1. **Upload** your invoice PDFs using the file uploader above
            2. **Click** "Process Invoices" to start extraction
            3. **Review** the extracted data in the results table
            4. **Download** as CSV or Excel for your records

            ### 🎯 What Gets Extracted

            | Field | Description |
            |-------|-------------|
            | **Vendor Name** | Company that sent the invoice |
            | **Business Name** | Company being billed |
            | **Invoice Number** | Document reference number |
            | **Amount** | Total amount due |
            | **Invoice Date** | Date on the invoice |
            | **Payment Terms** | e.g., "Net 30", "10 days" |
            | **Due Date** | When payment is due (calculated if needed) |
            | **Notes** | PAST DUE, PAID, or other status |
            """)


if __name__ == "__main__":
    main()