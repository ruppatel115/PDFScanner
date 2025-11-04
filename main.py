import streamlit as st
import os
import json
from datetime import datetime
import logging
import re
import traceback
import tempfile
from pathlib import Path
import zipfile
import io
import base64

# PDF and AI components
try:
    import PyPDF2
except ImportError:
    st.error("Please install PyPDF2: pip install PyPDF2")

try:
    from openai import OpenAI
except ImportError:
    st.error("Please install openai: pip install openai")

# Optional dependencies for OCR
OCR_AVAILABLE = False
try:
    import pytesseract
    from pdf2image import convert_from_path
    import cv2
    import numpy as np

    OCR_AVAILABLE = True
except ImportError as e:
    st.warning(f"OCR features disabled: {e}")

# Configure page
st.set_page_config(
    page_title="AI PDF Document Renamer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


class PDFInvoiceProcessor:
    def __init__(self):
        self.client = None
        self.setup_logging()
        self.vendor_normalization_cache = {}

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_openai_client(self, api_key):
        """Initialize OpenAI client with API key"""
        if api_key and api_key.startswith('sk-'):
            try:
                self.client = OpenAI(api_key=api_key)
                # Quick test to verify API key works
                try:
                    test_response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Say 'test'"}],
                        max_tokens=5
                    )
                    return True, "✅ OpenAI client initialized successfully"
                except Exception as test_error:
                    error_msg = str(test_error)
                    if "insufficient_quota" in error_msg:
                        return False, "❌ API key valid but insufficient quota"
                    elif "invalid_api_key" in error_msg:
                        return False, "❌ Invalid API key"
                    else:
                        return False, f"❌ API test failed: {error_msg}"
            except Exception as e:
                return False, f"❌ Error initializing OpenAI client: {str(e)}"
        else:
            return False, "⚠️ Please enter a valid OpenAI API key (should start with 'sk-')"

    def normalize_vendor_name(self, vendor_name):
        """Normalize vendor name to ensure consistent grouping"""
        if not vendor_name or vendor_name == 'UnknownVendor':
            return 'UnknownVendor'

        if vendor_name in self.vendor_normalization_cache:
            return self.vendor_normalization_cache[vendor_name]

        normalized = vendor_name.lower().strip()

        suffixes_to_remove = [
            r'\binc\.?$', r'\bllc\.?$', r'\bltd\.?$', r'\bcorp\.?$', r'\bcorporation\.?$',
            r'\bcompany\.?$', r'\bco\.?$', r'\bllp\.?$', r'\bplc\.?$', r'\bgmbh\.?$',
            r'\bincorporated\.?$', r'\blimited\.?$'
        ]

        for suffix in suffixes_to_remove:
            normalized = re.sub(suffix, '', normalized)

        normalized = re.sub(r'[^\w\s\-]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        if not normalized:
            normalized = vendor_name.lower().strip()

        normalized = normalized.title()
        self.vendor_normalization_cache[vendor_name] = normalized

        return normalized

    def extract_scan_date_from_filename(self, filename):
        """Extract scan date from filename"""
        try:
            date_pattern = r'(\d{4}-\d{2}-\d{2})'
            match = re.search(date_pattern, filename)
            if match:
                date_str = match.group(1)
                return datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y%m%d')
        except:
            pass
        return datetime.now().strftime('%Y%m%d')

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file using OCR for scanned documents"""
        try:
            # First try regular text extraction
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                if text.strip():
                    return text

            # If no text found, use OCR for scanned documents
            return self.extract_text_with_ocr(pdf_path)

        except Exception as e:
            return self.extract_text_with_ocr(pdf_path)

    def extract_text_with_ocr(self, pdf_path):
        """Extract text from scanned PDF using OCR"""
        if not OCR_AVAILABLE:
            return None

        try:
            images = convert_from_path(pdf_path, dpi=200)
            full_text = ""

            for image in images:
                try:
                    open_cv_image = np.array(image)
                    open_cv_image = open_cv_image[:, :, ::-1].copy()

                    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    page_text = pytesseract.image_to_string(thresh, config='--psm 6')
                    full_text += page_text + "\n"
                except Exception:
                    continue

            return full_text if full_text.strip() else None

        except Exception:
            return None

    def extract_account_number_with_regex(self, text):
        """Extract account numbers and important identifiers using focused regex patterns"""
        try:
            patterns = [
                r'account\s*[#:]?\s*([A-Z0-9\-]{5,})',
                r'acct\s*[#:]?\s*([A-Z0-9\-]{5,})',
                r'account\s*id\s*[#:]?\s*([A-Z0-9\-]{5,})',
                r'policy[/\s]*account\s*no\.?\s*[#:]?\s*([A-Z0-9\-]{5,})',
                r'policy\s*[#:]?\s*([A-Z0-9\-]{5,})',
                r'fein\s*[#:]?\s*([A-Z0-9\-]{5,})',
                r'invoice\s*[#:]?\s*([A-Z0-9\-]{3,})',
                r'letter\s*id\s*[#:]?\s*([A-Z0-9\-]{5,})',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    account_number = matches[0].strip()
                    if self.is_valid_account_number(account_number):
                        return account_number

            potential_numbers = re.findall(r'\b[A-Z0-9\-]{5,15}\b', text)
            for number in potential_numbers:
                if self.is_valid_account_number(number):
                    return number

            return "NoAcct"

        except Exception:
            return "NoAcct"

    def is_valid_account_number(self, number):
        """Validate if a string is likely to be an account number"""
        if not number or len(number) < 5:
            return False

        if re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$', number):
            return False

        if re.match(r'^\$?\d+\.\d{2}$', number):
            return False

        if re.match(r'^\d{7,}$', number):
            return False

        if re.match(r'^\d{6,9}$', number) and not any(keyword in number.lower() for keyword in ['check', 'chk']):
            return False

        return True

    def extract_business_name_with_regex(self, text):
        """Extract business names using focused patterns"""
        try:
            lines = text.split('\n')

            for i, line in enumerate(lines):
                line_clean = line.strip()

                if any(prefix in line_clean.lower() for prefix in
                       ['bill to', 'ship to', 'to:', 'payer name:', 'customer:']):
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line and len(next_line) > 3:
                            return next_line

                if i < 5 and len(line_clean) > 5 and not any(
                        word in line_clean.lower() for word in ['page', 'date', 'invoice', 'account']):
                    if (re.match(r'^[A-Za-z\s,&\.]+$', line_clean) and
                            len(line_clean) > 5 and
                            not re.match(r'^\d', line_clean) and
                            not re.match(r'.*\d{5,}.*', line_clean)):
                        return line_clean

            return "UnknownBusiness"

        except Exception:
            return "UnknownBusiness"

    def analyze_document_with_ai(self, text):
        """Use AI to extract business name and account number with specific focus"""
        if not self.client:
            return None

        text = text[:12000]

        prompt = f"""
        Analyze this document text and extract the following key information:

        BUSINESS NAME: Identify the MAIN BUSINESS or COMPANY that this document is for. This is typically:
        - The "Bill To" company
        - The "Ship To" company  
        - The account holder
        - The customer name
        - The payer name

        IMPORTANT: Look for the ACTUAL BUSINESS NAME, not the vendor/sender.

        ACCOUNT NUMBER: Extract the most important PERMANENT IDENTIFIER for this business:
        - Account numbers (look for "Account #", "Account ID", "Acct #")
        - Policy numbers (look for "Policy/Account No")
        - FEIN numbers (look for "FEIN")
        - Customer account numbers
        - Permanent reference numbers

        Return ONLY valid JSON with these exact keys: business_name, account_number

        CRITICAL GUIDELINES:
        - For business_name: Extract the ACTUAL BUSINESS/CUSTOMER name, not the vendor
        - For account_number: Look for permanent identifiers, NOT temporary numbers like invoice numbers or check numbers
        - If no clear business name found, use "UnknownBusiness"
        - If no clear account number found, use "NoAcct"

        Document text:
        {text}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You extract the business/customer name and account number from documents. Focus on the actual business the document is for, not the sender. Return valid JSON with business_name and account_number keys."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            result = response.choices[0].message.content.strip()
            result = re.sub(r'```json\s*|\s*```', '', result).strip()

            data = json.loads(result)

            if 'business_name' in data:
                original_business = data['business_name']
                data['business_name'] = self.normalize_vendor_name(original_business)
                data['original_business'] = original_business
            else:
                data['business_name'] = 'UnknownBusiness'

            if 'account_number' not in data:
                data['account_number'] = 'NoAcct'

            return data

        except Exception as e:
            print(f"AI Analysis Error: {e}")
            return {'business_name': 'UnknownBusiness', 'account_number': 'NoAcct'}

    def safe_filename(self, text):
        """Convert text to safe filename"""
        if not text:
            return "Unknown"
        safe = re.sub(r'[^\w\s\-\.]', '', str(text))
        safe = re.sub(r'\s+', ' ', safe)
        return safe.strip()

    def generate_filename(self, scan_date, document_data):
        """Generate filename in format: YYYYMMDD BusinessName AccountNumber.pdf"""
        try:
            business_name = document_data.get('business_name', 'UnknownBusiness')
            account_number = document_data.get('account_number', 'NoAcct')

            safe_business = self.safe_filename(business_name)
            safe_account = self.safe_filename(account_number)

            filename = f"{scan_date} {safe_business} {safe_account}.pdf"
            return filename

        except Exception as e:
            print(f"DEBUG: Error generating filename: {e}")
            return f"{scan_date} UnknownBusiness NoAcct.pdf"

    def process_single_pdf(self, file_content, original_filename):
        """Process a single PDF file and return the processed file data"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name

            try:
                # Extract scan date
                scan_date = self.extract_scan_date_from_filename(original_filename)

                # Extract text
                text = self.extract_text_from_pdf(tmp_path)
                if not text:
                    return False, f"❌ Could not extract text from: {original_filename}"

                # Use regex to extract business name and account number first
                regex_business = self.extract_business_name_with_regex(text)
                regex_account = self.extract_account_number_with_regex(text)

                # Analyze with AI
                document_data = self.analyze_document_with_ai(text)
                if not document_data:
                    return False, f"❌ AI analysis failed for: {original_filename}"

                # Use regex as fallback for business name and account number
                ai_business = document_data.get('business_name', 'UnknownBusiness')
                ai_account = document_data.get('account_number', 'NoAcct')

                # Improved business name selection
                if ai_business == 'UnknownBusiness' and regex_business != 'UnknownBusiness':
                    document_data['business_name'] = self.normalize_vendor_name(regex_business)
                    document_data['original_business'] = regex_business

                # Improved account number selection
                if ai_account == 'NoAcct' and regex_account != 'NoAcct':
                    document_data['account_number'] = regex_account

                # Generate new filename
                new_filename = self.generate_filename(scan_date, document_data)
                if not new_filename:
                    return False, f"❌ Failed to generate filename for: {original_filename}"

                # Return the processed file data
                return True, {
                    "original_filename": original_filename,
                    "new_filename": new_filename,
                    "business_name": document_data.get('business_name', 'UnknownBusiness'),
                    "original_business": document_data.get('original_business', 'UnknownBusiness'),
                    "account_number": document_data.get('account_number', 'NoAcct'),
                    "file_content": file_content,  # Keep original content for ZIP
                    "scan_date": scan_date
                }

            finally:
                # Always clean up the temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        except Exception as e:
            # Clean up temporary file if it exists
            try:
                if 'tmp_path' in locals():
                    os.unlink(tmp_path)
            except:
                pass
            return False, f"❌ Error processing {original_filename}: {str(e)}"


def create_zip_from_processed_files(processed_files):
    """Create a ZIP file in memory from processed files"""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Organize files by business name in the ZIP
        business_folders = {}

        for result in processed_files:
            if result['success']:
                file_data = result['data']
                business_name = file_data['business_name']
                safe_business = re.sub(r'[^\w\s\-]', '', business_name).replace(' ', '_')

                # Create business folder structure in ZIP
                zip_path = f"{safe_business}/{file_data['new_filename']}"

                # Add file to ZIP
                zip_file.writestr(zip_path, file_data['file_content'])

                # Track business folders for organization
                if safe_business not in business_folders:
                    business_folders[safe_business] = []
                business_folders[safe_business].append(file_data['new_filename'])

    zip_buffer.seek(0)
    return zip_buffer


def main():
    st.title("📄 AI PDF Document Renamer")
    st.markdown("### Secure Processing - Files Downloaded as ZIP, Nothing Saved Locally")

    # Initialize processor
    if 'processor' not in st.session_state:
        st.session_state.processor = PDFInvoiceProcessor()
        st.session_state.api_key_valid = False
        st.session_state.processing_results = []

    # Sidebar for configuration
    with st.sidebar:
        st.header("🔑 Configuration")

        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Get your API key from platform.openai.com",
            key="api_key_input"
        )

        if st.button("Validate API Key", key="validate_btn"):
            if api_key:
                with st.spinner("Validating API key..."):
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

        # Show current status
        if st.session_state.get('api_key_valid', False):
            st.success("✅ API Key Valid")
        else:
            st.warning("❌ API Key Not Valid")

        st.markdown("---")
        st.header("📋 Instructions")
        st.markdown("""
        1. 🔑 Enter & validate your OpenAI API key
        2. 📁 Upload PDF files using the file uploader
        3. 🚀 Click 'Process Files' to start
        4. 📥 **Automatic ZIP download** - files are NOT saved locally

        **Security Features:**
        - ✅ No local storage of processed files
        - ✅ All processing in temporary memory
        - ✅ Automatic cleanup after download
        - ✅ Files organized in ZIP by business name

        **Extraction Focus:**
        - Business names (who the document is FOR)
        - Account numbers (permanent identifiers)
        - OCR for scanned documents
        """)

    # Main content area
    tab1, tab2 = st.tabs(["📁 Upload & Process", "⚙️ Settings & Info"])

    with tab1:
        st.header("Upload PDF Files")

        if not st.session_state.get('api_key_valid', False):
            st.warning("⚠️ Please enter and validate a valid OpenAI API key in the sidebar to continue.")
        else:
            # File upload section
            st.subheader("📤 Upload Your PDF Files")

            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type="pdf",
                accept_multiple_files=True,
                help="Select one or more PDF files to process. Files will be downloaded as ZIP - nothing saved locally.",
                key="file_uploader"
            )

            if uploaded_files:
                st.success(f"📄 Selected {len(uploaded_files)} file(s) for processing")

                # Show file list
                with st.expander("View Selected Files", expanded=True):
                    for i, file in enumerate(uploaded_files):
                        st.write(f"{i + 1}. {file.name} ({file.size / 1024:.1f} KB)")

                # Processing button
                if st.button("🚀 Process & Download ZIP", type="primary", use_container_width=True):
                    processed_files = []
                    log_messages = []

                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, uploaded_file in enumerate(uploaded_files):
                        # Update progress
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        status_text.text(f"🔍 Processing {i + 1}/{len(uploaded_files)}: {uploaded_file.name}")

                        # Process file
                        success, result = st.session_state.processor.process_single_pdf(
                            uploaded_file.getvalue(),
                            uploaded_file.name
                        )

                        if success:
                            processed_files.append({
                                "success": True,
                                "data": result
                            })
                            log_messages.append(f"✅ {result['original_filename']} → {result['new_filename']}")
                        else:
                            processed_files.append({
                                "success": False,
                                "error": result
                            })
                            log_messages.append(result)

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()

                    # Create and download ZIP if we have successful processed files
                    successful_files = [p for p in processed_files if p['success']]

                    if successful_files:
                        st.success(
                            f"🎉 Successfully processed {len(successful_files)} out of {len(uploaded_files)} files!")

                        # Create ZIP in memory
                        with st.spinner("Creating ZIP file..."):
                            zip_buffer = create_zip_from_processed_files(successful_files)

                        # Show business grouping information
                        businesses = {}
                        for result in successful_files:
                            business = result['data']['business_name']
                            if business not in businesses:
                                businesses[business] = []
                            businesses[business].append(result['data'])

                        # Display business grouping
                        with st.expander("🏢 Business Grouping Summary", expanded=True):
                            for business, files in businesses.items():
                                st.write(f"**{business}**: {len(files)} file(s)")
                                for file in files:
                                    st.write(f"  - {file['original_filename']} → {file['new_filename']}")

                        # Auto-download the ZIP file
                        zip_filename = f"processed_documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

                        st.markdown("### 📥 Download Ready!")
                        st.download_button(
                            label="⬇️ Click to Download ZIP File",
                            data=zip_buffer,
                            file_name=zip_filename,
                            mime="application/zip",
                            use_container_width=True,
                            key="auto_download_zip"
                        )

                        # Show processing summary
                        with st.expander("📊 Processing Summary", expanded=True):
                            st.subheader("Processed Files")
                            for result in successful_files:
                                file_data = result['data']
                                col1, col2, col3 = st.columns([3, 2, 1])
                                with col1:
                                    st.write(f"**{file_data['original_filename']}**")
                                    if file_data.get('original_business') and file_data['original_business'] != \
                                            file_data['business_name']:
                                        st.caption(f"Original: {file_data['original_business']}")
                                with col2:
                                    st.write(f"→ **{file_data['new_filename']}**")
                                with col3:
                                    st.write(f"Acct: {file_data['account_number']}")

                        # Show detailed log
                        with st.expander("📋 Detailed Processing Log"):
                            for log in log_messages:
                                if log.startswith("✅"):
                                    st.success(log)
                                elif log.startswith("❌"):
                                    st.error(log)
                                else:
                                    st.info(log)
                    else:
                        st.error("❌ No files were successfully processed. Check the processing log for details.")

            else:
                # Show upload instructions when no files are selected
                st.info("""
                **💡 How to use:**
                1. Click 'Browse files' or drag & drop PDF files above
                2. Select one or multiple PDF files
                3. Click 'Process & Download ZIP' to start AI-powered renaming
                4. **Files automatically download as ZIP** - nothing saved locally

                **Security Guarantee:**
                - 🔒 No files stored on server
                - 🔒 All processing in temporary memory
                - 🔒 Automatic cleanup after download
                - 🔒 ZIP organized by business names

                **File Naming Format:**
                `YYYYMMDD BusinessName AccountNumber.pdf`
                """)

    with tab2:
        st.header("Settings & Information")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🛠️ System Status")
            if OCR_AVAILABLE:
                st.success("✅ OCR features are available")
            else:
                st.warning("⚠️ OCR features are disabled")

            if st.session_state.get('api_key_valid', False):
                st.success("✅ OpenAI API is connected")
            else:
                st.warning("⚠️ OpenAI API not configured")

        with col2:
            st.subheader("🔒 Security Features")
            st.success("✅ No local file storage")
            st.success("✅ In-memory processing only")
            st.success("✅ Automatic temporary file cleanup")
            st.success("✅ ZIP download only")

        st.subheader("🔧 Technical Information")
        st.info("""
        **Processing Flow:**
        1. Files uploaded to temporary memory
        2. AI extracts business names and account numbers
        3. Files renamed and organized in memory
        4. ZIP created with business folder structure
        5. ZIP downloaded to your computer
        6. **All temporary data automatically deleted**

        **What We Extract:**
        - **Business Names**: Bill To, Customer, Payer names
        - **Account Numbers**: Permanent identifiers only
        - **Policy Numbers**: Insurance policy numbers
        - **FEIN Numbers**: Tax identification numbers

        **No Local Storage:**
        - Files never written to disk
        - No database storage
        - No file system persistence
        - Complete privacy protection
        """)


if __name__ == "__main__":
    main()