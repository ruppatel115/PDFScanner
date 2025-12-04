import streamlit as st
import os
import json
from datetime import datetime
import logging
import shutil
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
        self.vendor_normalization_cache = {}  # Cache for normalized vendor names

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_openai_client(self, api_key):
        """Initialize OpenAI client with API key - Updated for v1.x"""
        if api_key and api_key.startswith('sk-'):
            try:
                # Simple initialization without proxies for v1.x
                self.client = OpenAI(api_key=api_key)
                # Test the client with a simple call
                try:
                    # Quick test to verify API key works
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

        # Check cache first
        if vendor_name in self.vendor_normalization_cache:
            return self.vendor_normalization_cache[vendor_name]

        # Convert to lowercase for case-insensitive comparison
        normalized = vendor_name.lower().strip()

        # Remove common business suffixes and legal entities
        suffixes_to_remove = [
            r'\binc\.?$', r'\bllc\.?$', r'\bltd\.?$', r'\bcorp\.?$', r'\bcorporation\.?$',
            r'\bcompany\.?$', r'\bco\.?$', r'\bllp\.?$', r'\bplc\.?$', r'\bgmbh\.?$',
            r'\bincorporated\.?$', r'\blimited\.?$'
        ]

        for suffix in suffixes_to_remove:
            normalized = re.sub(suffix, '', normalized)

        # Remove punctuation and special characters (keep spaces, hyphens)
        normalized = re.sub(r'[^\w\s\-]', '', normalized)

        # Remove extra spaces and trim
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        # If we ended up with empty string, use original
        if not normalized:
            normalized = vendor_name.lower().strip()

        # Title case for consistency in display
        normalized = normalized.title()

        # Cache the result
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
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=200)
            full_text = ""

            for image in images:
                try:
                    # Convert PIL image to OpenCV format
                    open_cv_image = np.array(image)
                    open_cv_image = open_cv_image[:, :, ::-1].copy()

                    # Preprocess image for better OCR
                    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # Use pytesseract to extract text
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
            # Focus on specific patterns for account numbers and important IDs
            patterns = [
                # Account numbers with labels (most important)
                r'account\s*[#:]?\s*([A-Z0-9\-]{5,})',
                r'acct\s*[#:]?\s*([A-Z0-9\-]{5,})',
                r'account\s*id\s*[#:]?\s*([A-Z0-9\-]{5,})',

                # Policy numbers (for insurance documents)
                r'policy[/\s]*account\s*no\.?\s*[#:]?\s*([A-Z0-9\-]{5,})',
                r'policy\s*[#:]?\s*([A-Z0-9\-]{5,})',

                # FEIN numbers (tax IDs)
                r'fein\s*[#:]?\s*([A-Z0-9\-]{5,})',

                # Invoice numbers (only if clearly labeled)
                r'invoice\s*[#:]?\s*([A-Z0-9\-]{3,})',

                # Letter IDs for official documents
                r'letter\s*id\s*[#:]?\s*([A-Z0-9\-]{5,})',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    account_number = matches[0].strip()
                    # Validate it's a reasonable account number (not a date, amount, etc.)
                    if self.is_valid_account_number(account_number):
                        return account_number

            # Look for standalone numbers that look like account numbers
            # Focus on numbers that are 5+ digits and not dates/amounts
            potential_numbers = re.findall(r'\b[A-Z0-9\-]{5,15}\b', text)
            for number in potential_numbers:
                if self.is_valid_account_number(number):
                    return number

            return "NoAcct"

        except Exception:
            return "NoAcct"

    def is_valid_account_number(self, number):
        """Validate if a string is likely to be an account number"""
        # Remove common non-account number patterns
        if not number or len(number) < 5:
            return False

        # Exclude dates
        if re.match(r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$', number):
            return False

        # Exclude amounts with decimals
        if re.match(r'^\$?\d+\.\d{2}$', number):
            return False

        # Exclude pure numbers that are too long (like meter readings)
        if re.match(r'^\d{7,}$', number):
            return False

        # Exclude check numbers (typically 6-9 digits)
        if re.match(r'^\d{6,9}$', number) and not any(keyword in number.lower() for keyword in ['check', 'chk']):
            return False

        return True

    def extract_business_name_with_regex(self, text):
        """Extract business names using focused patterns"""
        try:
            lines = text.split('\n')

            # Look for common business name patterns
            for i, line in enumerate(lines):
                line_clean = line.strip()

                # Look for company names in bill to/ship to sections
                if any(prefix in line_clean.lower() for prefix in
                       ['bill to', 'ship to', 'to:', 'payer name:', 'customer:']):
                    # Next line often contains the business name
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line and len(next_line) > 3:
                            return next_line

                # Look for company names at the top of documents
                if i < 5 and len(line_clean) > 5 and not any(
                        word in line_clean.lower() for word in ['page', 'date', 'invoice', 'account']):
                    # Check if it looks like a company name (not an address line, etc.)
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

        IMPORTANT: Look for the ACTUAL BUSINESS NAME, not the vendor/sender. For example:
        - If it's a utility bill, the business name is who is being billed
        - If it's an insurance document, the business name is the policy holder
        - If it's a tax document, the business name is the taxpayer

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

            # Apply normalization to business name for consistency
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

            print(f"DEBUG: Generated filename: {filename}")
            return filename

        except Exception as e:
            print(f"DEBUG: Error generating filename: {e}")
            return f"{scan_date} UnknownBusiness NoAcct.pdf"

    def process_single_pdf(self, file_content, original_filename, output_base):
        """Process a single PDF file with focused business name and account number extraction"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name

            # Extract scan date
            scan_date = self.extract_scan_date_from_filename(original_filename)

            # Extract text
            text = self.extract_text_from_pdf(tmp_path)
            if not text:
                os.unlink(tmp_path)
                return False, f"❌ Could not extract text from: {original_filename}"

            # Use regex to extract business name and account number first
            regex_business = self.extract_business_name_with_regex(text)
            regex_account = self.extract_account_number_with_regex(text)

            print(f"DEBUG: Regex found - Business: {regex_business}, Account: {regex_account}")

            # Analyze with AI
            document_data = self.analyze_document_with_ai(text)
            if not document_data:
                os.unlink(tmp_path)
                return False, f"❌ AI analysis failed for: {original_filename}"

            # Use regex as fallback for business name and account number
            ai_business = document_data.get('business_name', 'UnknownBusiness')
            ai_account = document_data.get('account_number', 'NoAcct')

            print(f"DEBUG: AI found - Business: {ai_business}, Account: {ai_account}")

            # Improved business name selection
            if ai_business == 'UnknownBusiness' and regex_business != 'UnknownBusiness':
                document_data['business_name'] = self.normalize_vendor_name(regex_business)
                document_data['original_business'] = regex_business
                print(f"DEBUG: Using regex business name: {regex_business}")

            # Improved account number selection
            if ai_account == 'NoAcct' and regex_account != 'NoAcct':
                document_data['account_number'] = regex_account
                print(f"DEBUG: Using regex account number: {regex_account}")

            # Generate new filename
            new_filename = self.generate_filename(scan_date, document_data)
            if not new_filename:
                os.unlink(tmp_path)
                return False, f"❌ Failed to generate filename for: {original_filename}"

            # Create business folder structure using NORMALIZED business name
            business_name = document_data.get('business_name', 'UnknownBusiness')
            original_business = document_data.get('original_business', business_name)
            safe_business_name = self.safe_filename(business_name).replace(' ', '_')
            business_folder = os.path.join(output_base, safe_business_name)
            os.makedirs(business_folder, exist_ok=True)

            # Copy file to organized location
            target_path = os.path.join(business_folder, new_filename)

            # Handle duplicates
            base, ext = os.path.splitext(target_path)
            counter = 1
            while os.path.exists(target_path):
                target_path = f"{base}_{counter}{ext}"
                counter += 1

            shutil.copy2(tmp_path, target_path)

            # Clean up temporary file
            os.unlink(tmp_path)

            return True, {
                "original": original_filename,
                "new": new_filename,
                "business": business_name,
                "original_business": original_business,
                "account_number": document_data.get('account_number', 'NoAcct'),
                "path": target_path
            }

        except Exception as e:
            # Clean up temporary file if it exists
            try:
                if 'tmp_path' in locals():
                    os.unlink(tmp_path)
            except:
                pass
            return False, f"❌ Error processing {original_filename}: {str(e)}"


def main():
    st.title("📄 AI PDF Document Renamer")
    st.markdown("### Focused Business Name & Account Number Extraction")

    # Initialize processor
    if 'processor' not in st.session_state:
        st.session_state.processor = PDFInvoiceProcessor()
        st.session_state.api_key_valid = False
        st.session_state.processed_files = []
        st.session_state.processing_log = []

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
        4. 📥 Download processed files as ZIP

        **Features:**
        - **Business name extraction** (who the document is FOR)
        - **Account number extraction** (permanent identifiers)
        - OCR for scanned documents
        - Organized folder structure by business
        - Consistent business grouping
        """)

    # Main content area
    tab1, tab2 = st.tabs(["📁 Upload & Process", "⚙️ Settings & Info"])

    with tab1:
        st.header("Upload PDF Files")

        if not st.session_state.get('api_key_valid', False):
            st.warning("⚠️ Please enter and validate a valid OpenAI API key in the sidebar to continue.")
        else:
            # File upload section with better styling
            st.subheader("📤 Upload Your PDF Files")

            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type="pdf",
                accept_multiple_files=True,
                help="Select one or more PDF files to process. You can select multiple files at once.",
                key="file_uploader"
            )

            if uploaded_files:
                st.success(f"📄 Selected {len(uploaded_files)} file(s) for processing")

                # Show file list
                with st.expander("View Selected Files", expanded=True):
                    for i, file in enumerate(uploaded_files):
                        st.write(f"{i + 1}. {file.name} ({file.size / 1024:.1f} KB)")

                # Processing options
                col1, col2 = st.columns([1, 2])
                with col1:
                    if st.button("🚀 Process Files", type="primary", use_container_width=True):
                        # Create temporary directory for processing
                        with tempfile.TemporaryDirectory() as temp_dir:
                            processed_files = []
                            log_messages = []

                            # Progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            results_placeholder = st.empty()

                            for i, uploaded_file in enumerate(uploaded_files):
                                # Update progress
                                progress = (i + 1) / len(uploaded_files)
                                progress_bar.progress(progress)
                                status_text.text(f"🔍 Processing {i + 1}/{len(uploaded_files)}: {uploaded_file.name}")

                                # Process file
                                success, result = st.session_state.processor.process_single_pdf(
                                    uploaded_file.getvalue(),
                                    uploaded_file.name,
                                    temp_dir
                                )

                                if success:
                                    processed_files.append(result)
                                    log_messages.append(f"✅ {result['original']} → {result['new']}")
                                    st.session_state.processed_files.append(result)
                                else:
                                    log_messages.append(result)
                                    st.session_state.processing_log.append(result)

                            # Create ZIP file if we have processed files
                            if processed_files:
                                zip_buffer = io.BytesIO()
                                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                    for root, dirs, files in os.walk(temp_dir):
                                        for file in files:
                                            file_path = os.path.join(root, file)
                                            arcname = os.path.relpath(file_path, temp_dir)
                                            zip_file.write(file_path, arcname)

                                zip_buffer.seek(0)

                                # Show success message and download button
                                st.success(
                                    f"🎉 Successfully processed {len(processed_files)} out of {len(uploaded_files)} files!")

                                # Show business grouping information
                                businesses = {}
                                for result in processed_files:
                                    business = result['business']
                                    if business not in businesses:
                                        businesses[business] = []
                                    businesses[business].append(result)

                                # Display business grouping
                                with st.expander("🏢 Business Grouping Summary", expanded=True):
                                    for business, files in businesses.items():
                                        st.write(f"**{business}**: {len(files)} file(s)")
                                        for file in files:
                                            st.write(f"  - {file['original']} → {file['new']}")

                                # Download button
                                st.download_button(
                                    label="📥 Download Processed Files (ZIP)",
                                    data=zip_buffer,
                                    file_name="processed_documents.zip",
                                    mime="application/zip",
                                    use_container_width=True
                                )

                                # Show processing summary
                                with st.expander("📊 Processing Summary", expanded=True):
                                    st.subheader("Processed Files")
                                    for result in processed_files:
                                        col1, col2, col3 = st.columns([3, 2, 1])
                                        with col1:
                                            st.write(f"**{result['original']}**")
                                            if result.get('original_business') and result['original_business'] != \
                                                    result['business']:
                                                st.caption(f"Original: {result['original_business']}")
                                        with col2:
                                            st.write(f"→ **{result['new']}**")
                                        with col3:
                                            st.write(f"Acct: {result['account_number']}")

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
                                st.error(
                                    "❌ No files were successfully processed. Check the processing log for details.")

                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()

                with col2:
                    if st.button("🗑️ Clear Files", use_container_width=True):
                        st.session_state.processed_files = []
                        st.session_state.processing_log = []
                        st.rerun()

            else:
                # Show upload instructions when no files are selected
                st.info("""
                **💡 How to use:**
                1. Click 'Browse files' or drag & drop PDF files above
                2. Select one or multiple PDF files
                3. Click 'Process Files' to start AI-powered renaming
                4. Download the organized ZIP file

                **FOCUSED EXTRACTION:**
                - **Business Names**: Extracts who the document is FOR (Bill To, Customer, Payer)
                - **Account Numbers**: Finds permanent identifiers (Account #, Policy #, FEIN)
                - **Smart Filtering**: Ignores temporary numbers (check numbers, invoice numbers)
                - **Consistent Grouping**: Same business = same folder
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
            st.subheader("📊 Statistics")
            if st.session_state.processed_files:
                st.write(f"📁 Files processed: {len(st.session_state.processed_files)}")

                # Show business count
                businesses = set(f['business'] for f in st.session_state.processed_files)
                st.write(f"🏢 Unique businesses: {len(businesses)}")

                # Show account numbers found
                account_numbers = [f['account_number'] for f in st.session_state.processed_files if
                                   f['account_number'] != 'NoAcct']
                st.write(
                    f"🔢 Documents with account numbers: {len(account_numbers)}/{len(st.session_state.processed_files)}")
            else:
                st.write("No files processed yet")

        st.subheader("🔧 Technical Information")
        st.info("""
        **Supported Features:**
        - PDF text extraction using PyPDF2
        - OCR for scanned documents (Tesseract)
        - AI-powered business name and account number detection
        - Automatic file organization by business
        - Duplicate file handling
        - ZIP file export
        - Business name normalization
        - **Focused account number extraction**

        **What We Extract:**
        - **Business Names**: Bill To, Customer, Payer names
        - **Account Numbers**: Permanent identifiers only
        - **Policy Numbers**: Insurance policy numbers
        - **FEIN Numbers**: Tax identification numbers
        - **Account IDs**: Customer account numbers

        **File Naming Format:**
        `YYYYMMDD BusinessName AccountNumber.pdf`

        **Output Structure:**
        - Files organized in business-named folders
        - Original files preserved
        - Automatic duplicate resolution
        """)


if __name__ == "__main__":
    main()