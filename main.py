import streamlit as st

# MUST be first Streamlit command - before any other st.* calls
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

# AI integration
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

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
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_openai_client(self, api_key):
        """Initialize OpenAI client with API key"""
        if not OPENAI_AVAILABLE:
            return False, "❌ OpenAI library not installed. Run: pip install openai"

        if api_key and (api_key.startswith('sk-') or api_key.startswith('sk-proj-')):
            try:
                self.client = OpenAI(api_key=api_key)
                test_response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                return True, "✅ OpenAI API initialized successfully"

            except Exception as e:
                error_msg = str(e).lower()
                if "insufficient_quota" in error_msg:
                    return False, "❌ API key valid but insufficient quota"
                elif "invalid_api_key" in error_msg or "auth" in error_msg:
                    return False, "❌ Invalid API key"
                else:
                    return False, f"❌ API error: {str(e)[:100]}"
        else:
            return False, "⚠️ Please enter a valid OpenAI API key (should start with 'sk-')"

    def pdf_to_base64_image(self, pdf_path):
        """Convert first page of PDF to base64 image for vision API"""
        if not PDF2IMAGE_AVAILABLE or not PIL_AVAILABLE:
            return None

        try:
            images = convert_from_path(pdf_path, dpi=250, first_page=1, last_page=1)
            if images:
                img = images[0]
                buffered = io.BytesIO()
                img.save(buffered, format="PNG", optimize=True)
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                return img_base64
        except Exception as e:
            self.logger.error(f"Error converting PDF to image: {e}")
        return None

    def extract_invoice_data_with_vision(self, pdf_path, filename):
        """Use GPT-4 Vision to extract invoice data from PDF image"""
        if not self.client:
            return None

        img_base64 = self.pdf_to_base64_image(pdf_path)
        if not img_base64:
            self.logger.warning("Could not convert PDF to image")
            return None

        prompt = """You are a precise document data extractor. Examine this document carefully and extract ONLY what you can clearly see.

RETURN THIS EXACT JSON FORMAT (use null for any field you cannot find or are unsure about):
{
    "vendor_name": "Company/organization that ISSUED this document (in letterhead/header)",
    "business_name": "Company/person RECEIVING this document (Bill To, To, Customer)",
    "invoice_number": "Document reference number (Invoice #, Letter ID, Check No, etc.)",
    "amount": 0.00,
    "invoice_date": "YYYY-MM-DD",
    "payment_terms": "Payment terms if shown (e.g., '10 days', 'Net 30', 'Due on Receipt')",
    "due_date": "YYYY-MM-DD",
    "notes": "Any warnings, stamps, or status indicators (e.g., 'PAST DUE', 'PAID', 'FINAL NOTICE')"
}

EXTRACTION RULES:

1. VENDOR_NAME: The organization at the TOP/HEADER of the document - who SENT it
   - Look for company name in letterhead, logo area, or return address

2. BUSINESS_NAME: The recipient of this document
   - Look for "Bill To:", "To:", "Customer:", "Ship To:" sections
   - This is who the document is addressed to

3. INVOICE_NUMBER: The document's unique identifier
   - Common labels: "Invoice #", "Invoice Number", "Inv #", "Letter ID", "Document #", "Check No.", "Account ID"
   - Extract the number/code shown

4. AMOUNT: The TOTAL amount due - look at the BOTTOM of the document
   - Look for: "Total", "Amount Due", "Balance Due", "Total Due", "Payment Amount"
   - Extract as a number WITHOUT currency symbols (e.g., 520.91 not $520.91)
   - If multiple amounts shown, use the FINAL TOTAL at the bottom

5. INVOICE_DATE: The date the document was created/issued
   - Look for: "Date", "Invoice Date", "Date Issued", "Document Date"
   - Format as YYYY-MM-DD

6. PAYMENT_TERMS: How long to pay
   - Look for: "Terms", "Payment Terms", "Net" followed by a number
   - Examples: "10 days", "Net 30", "Due on Receipt"

7. DUE_DATE: When payment is due - CALCULATE if not explicitly shown
   - If "Due Date" is explicitly shown, use that date
   - If only Terms shown (e.g., "10 days") and you have Invoice Date, CALCULATE: Invoice Date + Terms = Due Date
   - Example: Invoice Date 5/31/2025 + Terms "10 days" = Due Date 2025-06-10
   - Format as YYYY-MM-DD
   - If you cannot determine, use null

8. NOTES: Any special status indicators or warnings visible on the document
   - Look for stamps, watermarks, or prominent text like: "PAST DUE", "PAID", "FINAL NOTICE", "OVERDUE", "URGENT", "NON-NEGOTIABLE"
   - Also note document type if not a standard invoice: "Check", "Notice", "Statement"
   - If none found, use null

IMPORTANT:
- Be PRECISE - only extract what you can clearly read
- Use null for any field you're unsure about - do NOT guess
- For amounts, ensure you're getting the TOTAL, not line items
- Always try to calculate due_date from invoice_date + payment_terms if due_date is not explicitly shown"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert document analyst. Extract data precisely and use null when uncertain. Always return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.0,  # Most deterministic
                max_tokens=600
            )

            result = response.choices[0].message.content.strip()
            result = re.sub(r'```json\s*|\s*```', '', result).strip()

            self.logger.info(f"Vision API response for {filename}: {result[:300]}")

            data = json.loads(result)
            return self._validate_and_clean_data(data)

        except Exception as e:
            self.logger.error(f"Vision API extraction error: {e}")
            return None

    def _validate_and_clean_data(self, data):
        """Validate and clean extracted data"""

        # Text fields - use None/empty if not found
        if not data.get('vendor_name'):
            data['vendor_name'] = None
        if not data.get('business_name'):
            data['business_name'] = None
        if not data.get('invoice_number'):
            data['invoice_number'] = None
        if not data.get('payment_terms'):
            data['payment_terms'] = None
        if not data.get('notes'):
            data['notes'] = None

        # Handle amount
        if data.get('amount') is not None:
            try:
                amount_str = str(data['amount']).replace('$', '').replace(',', '').strip()
                data['amount'] = float(amount_str)
            except:
                data['amount'] = None
        else:
            data['amount'] = None

        # Validate invoice_date format
        if data.get('invoice_date'):
            try:
                parsed_date = pd.to_datetime(data['invoice_date'])
                data['invoice_date'] = parsed_date.strftime('%Y-%m-%d')
            except:
                data['invoice_date'] = None
        else:
            data['invoice_date'] = None

        # Validate due_date format
        if data.get('due_date'):
            try:
                parsed_date = pd.to_datetime(data['due_date'])
                data['due_date'] = parsed_date.strftime('%Y-%m-%d')
            except:
                data['due_date'] = None
        else:
            data['due_date'] = None

        # If we have invoice_date and payment_terms but no due_date, try to calculate
        if data['invoice_date'] and data['payment_terms'] and not data['due_date']:
            data['due_date'] = self._calculate_due_date(data['invoice_date'], data['payment_terms'])

        return data

    def _calculate_due_date(self, invoice_date_str, terms):
        """Calculate due date from invoice date and payment terms"""
        try:
            invoice_date = datetime.strptime(invoice_date_str, '%Y-%m-%d')

            # Extract number of days from terms
            terms_lower = terms.lower()

            # Match patterns like "10 days", "net 30", "30 days", etc.
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
                self.logger.info(f"Processing {filename} with vision API")
                invoice_data = self.extract_invoice_data_with_vision(tmp_path, filename)

            os.unlink(tmp_path)

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
        st.error("❌ pypdf is not installed. Please run: `pip install pypdf`")
        st.stop()

    if not OPENAI_AVAILABLE:
        st.error("❌ openai is not installed. Please run: `pip install openai`")
        st.stop()

    st.title("📊 Invoice Data Extractor v3")
    st.markdown("### Extract structured data from invoices using GPT-4 Vision")

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
        st.header("🔑 Configuration")

        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Get your API key from platform.openai.com"
        )

        if st.button("Validate API Key"):
            if api_key:
                with st.spinner("Validating..."):
                    success, message = st.session_state.processor.initialize_openai_client(api_key)
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
            st.success("✅ API Key Valid")
        else:
            st.warning("⚠️ API Key Not Configured")

        st.markdown("---")
        st.header("🔧 System Status")
        if PDF2IMAGE_AVAILABLE:
            st.success("✅ Vision API Ready")
        else:
            st.error("❌ Install pdf2image: `pip install pdf2image`")

        st.markdown("---")
        st.header("📋 Extracted Fields")
        st.markdown("""
        - **Vendor Name** - Document issuer
        - **Business Name** - Recipient  
        - **Invoice Number** - Document ID
        - **Amount** - Total due
        - **Invoice Date** - Document date
        - **Payment Terms** - e.g., "10 days"
        - **Due Date** - Calculated/explicit
        - **Notes** - PAST DUE, PAID, etc.
        """)

    # Main content
    if not st.session_state.get('api_key_valid', False):
        st.warning("⚠️ Please configure your OpenAI API key in the sidebar.")
        st.info("""
        **Requirements:**
        ```bash
        pip install openai pypdf pdf2image pillow
        # Plus poppler:
        # Linux: apt-get install poppler-utils
        # Mac: brew install poppler
        ```
        """)
        st.stop()

    # Show results
    if st.session_state.results_df is not None and st.session_state.processing_complete:
        st.header("✅ Processing Complete!")

        df = st.session_state.results_df

        # Summary
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
            past_due = len(df[df['notes'].str.contains('PAST DUE', case=False, na=False)])
            st.metric("Past Due", past_due)

        # Data table
        st.subheader("📋 Extracted Data")

        # Reorder columns for display
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
                "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                "invoice_date": st.column_config.DateColumn("Invoice Date", format="YYYY-MM-DD"),
                "due_date": st.column_config.DateColumn("Due Date", format="YYYY-MM-DD"),
                "notes": st.column_config.TextColumn("Notes/Status", width="medium"),
            }
        )

        # Download buttons
        col1, col2 = st.columns(2)

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
            if st.button("🔄 Process New Files", use_container_width=True):
                st.session_state.results_df = None
                st.session_state.processing_complete = False
                st.rerun()

        # Breakdown sections
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 By Vendor")
            if df['vendor_name'].notna().any():
                vendor_summary = df[df['vendor_name'].notna()].groupby('vendor_name').agg({
                    'amount': 'sum',
                    'filename': 'count'
                }).rename(columns={'filename': 'count'}).sort_values('amount', ascending=False)
                st.dataframe(vendor_summary, column_config={
                    "amount": st.column_config.NumberColumn("Total", format="$%.2f"),
                    "count": "Count"
                })

        with col2:
            st.subheader("⚠️ Items Needing Attention")
            attention_df = df[
                (df['notes'].notna()) |
                (df['amount'].isna()) |
                (df['due_date'].isna())
                ][['filename', 'notes', 'amount', 'due_date']]
            if len(attention_df) > 0:
                st.dataframe(attention_df, hide_index=True)
            else:
                st.success("All items extracted successfully!")

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
                accept_multiple_files=True
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
                type="zip"
            )
            if zip_file:
                try:
                    with zipfile.ZipFile(io.BytesIO(zip_file.read())) as z:
                        pdf_files = [f for f in z.namelist() if
                                     f.lower().endswith('.pdf') and not f.startswith('__MACOSX')]
                        st.success(f"Found {len(pdf_files)} PDF files")
                        for pdf_name in pdf_files:
                            files_to_process.append({
                                'name': os.path.basename(pdf_name),
                                'content': z.read(pdf_name)
                            })
                except Exception as e:
                    st.error(f"Error reading ZIP: {e}")

        if files_to_process:
            st.success(f"📄 Ready to process {len(files_to_process)} file(s)")

            with st.expander("View files"):
                for i, file in enumerate(files_to_process):
                    st.write(f"{i + 1}. {file['name']}")

            if st.button("🚀 Process Invoices", type="primary", use_container_width=True):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, file in enumerate(files_to_process):
                    progress_bar.progress((i + 1) / len(files_to_process))
                    status_text.text(f"🔍 Processing {i + 1}/{len(files_to_process)}: {file['name']}")
                    result = st.session_state.processor.process_single_pdf(file['content'], file['name'])
                    results.append(result)

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
            st.info("""
            **💡 New in v3:**
            - **Invoice Date** column - the date shown on the document
            - **Payment Terms** column - e.g., "10 days", "Net 30"
            - **Due Date** is now CALCULATED from Invoice Date + Terms when not explicit
            - **Notes** column captures "PAST DUE", "PAID", "FINAL NOTICE", etc.
            - More conservative extraction - leaves fields blank when uncertain

            **Upload your PDFs to get started!**
            """)


if __name__ == "__main__":
    main()