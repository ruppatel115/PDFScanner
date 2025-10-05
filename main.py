import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import json
from datetime import datetime
import threading
import logging
from pathlib import Path
import shutil
import re
import sys
import traceback

# PDF and AI components with better error handling
try:
    import PyPDF2
except ImportError:
    print("Please install PyPDF2: pip install PyPDF2")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    sys.exit(1)

# Optional dependencies for OCR
OCR_AVAILABLE = False
try:
    import pytesseract
    from pdf2image import convert_from_path
    import cv2
    import numpy as np
    OCR_AVAILABLE = True
except ImportError as e:
    print(f"OCR features disabled: {e}")

# Configure customtkinter appearance
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class PDFInvoiceRenamer:
    def __init__(self, root):
        self.root = root
        self.root.title("AI PDF Invoice Renamer")
        self.root.geometry("1000x700")
        self.root.minsize(900, 650)

        # Configuration
        self.config_file = "config.json"
        self.load_config()

        # Set up UI first
        self.setup_ui()
        self.setup_logging()

        # Initialize OpenAI client only if API key is available
        self.client = None
        self.initialize_openai_client()

        # Monitoring
        self.observer = None
        self.monitoring = False

        # Track processing state
        self.is_processing = False

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('invoice_renamer.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_openai_client(self):
        """Initialize OpenAI client with current API key"""
        api_key = self.config.get('openai_api_key', '').strip()
        if api_key and api_key.startswith('sk-'):
            try:
                self.client = OpenAI(api_key=api_key)
                self.log("✅ OpenAI client initialized successfully")
                return True
            except Exception as e:
                self.log(f"❌ Error initializing OpenAI client: {str(e)}")
                self.client = None
                return False
        else:
            self.log("⚠️ No valid API key configured")
            return False

    def load_config(self):
        default_config = {
            'watch_folder': '',
            'output_folder': '',
            'openai_api_key': '',
            'model': 'gpt-3.5-turbo'
        }

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    self.config = {**default_config, **loaded_config}
            except Exception as e:
                self.log(f"Error loading config: {e}, using defaults")
                self.config = default_config
        else:
            self.config = default_config

    def save_config(self):
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            self.log(f"Error saving config: {e}")
            return False

    def setup_ui(self):
        # Create main container with sidebar and content area
        self.main_container = ctk.CTkFrame(self.root, corner_radius=0)
        self.main_container.pack(fill="both", expand=True)

        # Create sidebar
        self.sidebar = ctk.CTkFrame(self.main_container, width=200, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)

        # Create content area
        self.content_area = ctk.CTkFrame(self.main_container, corner_radius=0)
        self.content_area.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        self.setup_sidebar()
        self.setup_content_area()

    def setup_sidebar(self):
        # Logo/Title
        title_label = ctk.CTkLabel(
            self.sidebar,
            text="PDF Renamer",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.pack(pady=(30, 10))

        subtitle_label = ctk.CTkLabel(
            self.sidebar,
            text="AI-Powered Document Processing",
            font=ctk.CTkFont(size=12),
            text_color="gray70"
        )
        subtitle_label.pack(pady=(0, 30))

        # Navigation buttons
        nav_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        nav_frame.pack(fill="x", padx=15, pady=10)

        self.config_btn = ctk.CTkButton(
            nav_frame,
            text="⚙️ Configuration",
            command=self.show_configuration,
            anchor="w",
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.config_btn.pack(fill="x", pady=5)

        self.process_btn = ctk.CTkButton(
            nav_frame,
            text="📁 Process Files",
            command=self.show_process_files,
            anchor="w",
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.process_btn.pack(fill="x", pady=5)

        # Status indicator
        status_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        status_frame.pack(side="bottom", fill="x", padx=15, pady=20)

        self.status_label = ctk.CTkLabel(
            status_frame,
            text="🔴 Not Ready",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.status_label.pack()

        # Appearance mode
        appearance_label = ctk.CTkLabel(
            status_frame,
            text="Appearance Mode:",
            font=ctk.CTkFont(size=12)
        )
        appearance_label.pack(pady=(10, 5))

        appearance_option = ctk.CTkOptionMenu(
            status_frame,
            values=["Light", "Dark", "System"],
            command=self.change_appearance_mode,
            width=120
        )
        appearance_option.pack()
        appearance_option.set("System")

    def setup_content_area(self):
        # Create tab view for different sections
        self.tabview = ctk.CTkTabview(self.content_area, width=600)
        self.tabview.pack(fill="both", expand=True, padx=0, pady=0)

        # Create tabs
        self.config_tab = self.tabview.add("Configuration")
        self.process_tab = self.tabview.add("Process Files")
        self.log_tab = self.tabview.add("Activity Log")

        self.setup_config_tab()
        self.setup_process_tab()
        self.setup_log_tab()

    def setup_config_tab(self):
        # API Key Section
        api_frame = ctk.CTkFrame(self.config_tab)
        api_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            api_frame,
            text="OpenAI API Configuration",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))

        api_help = ctk.CTkLabel(
            api_frame,
            text="Get your API key from: platform.openai.com → API Keys",
            font=ctk.CTkFont(size=12),
            text_color="gray70"
        )
        api_help.pack(anchor="w", padx=15, pady=(0, 10))

        api_input_frame = ctk.CTkFrame(api_frame, fg_color="transparent")
        api_input_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(api_input_frame, text="API Key:").pack(side="left")

        self.api_key_var = ctk.StringVar(value=self.config.get('openai_api_key', ''))
        self.api_entry = ctk.CTkEntry(
            api_input_frame,
            textvariable=self.api_key_var,
            show="•",
            width=400,
            placeholder_text="sk-..."
        )
        self.api_entry.pack(side="left", padx=10)

        ctk.CTkButton(
            api_input_frame,
            text="Test Key",
            command=self.test_api_key,
            width=80
        ).pack(side="left", padx=5)

        # Output Folder Section
        output_frame = ctk.CTkFrame(self.config_tab)
        output_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(
            output_frame,
            text="Output Location",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))

        output_input_frame = ctk.CTkFrame(output_frame, fg_color="transparent")
        output_input_frame.pack(fill="x", padx=15, pady=5)

        ctk.CTkLabel(output_input_frame, text="Output Folder:").pack(side="left")
        self.output_folder_var = ctk.StringVar(value=self.config.get('output_folder', ''))
        self.output_entry = ctk.CTkEntry(
            output_input_frame,
            textvariable=self.output_folder_var,
            width=300
        )
        self.output_entry.pack(side="left", padx=10)
        ctk.CTkButton(
            output_input_frame,
            text="Browse",
            command=self.browse_output_folder,
            width=80
        ).pack(side="left", padx=5)

        # Save Button
        save_frame = ctk.CTkFrame(self.config_tab, fg_color="transparent")
        save_frame.pack(fill="x", padx=10, pady=20)

        ctk.CTkButton(
            save_frame,
            text="💾 Save Configuration",
            command=self.save_configuration,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=10)

    def setup_process_tab(self):
        # Main processing area
        area_frame = ctk.CTkFrame(self.process_tab)
        area_frame.pack(fill="both", expand=True, padx=20, pady=20)

        ctk.CTkLabel(
            area_frame,
            text="Upload PDF Files or Folder",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(30, 10))

        # Instructions
        instructions = ctk.CTkLabel(
            area_frame,
            text="• Select individual PDF files or an entire folder\n• Files will be renamed: YYYYMMDD VendorName $Amount.pdf\n• Organized into folders by vendor name\n• Supports both text-based and scanned PDFs",
            font=ctk.CTkFont(size=12),
            justify="left"
        )
        instructions.pack(pady=10)

        # Upload area
        self.drop_area = ctk.CTkFrame(area_frame, height=200, fg_color="#f0f0f0", corner_radius=10)
        self.drop_area.pack(fill="both", expand=False, padx=40, pady=20)

        drop_label = ctk.CTkLabel(
            self.drop_area, 
            text="📁 Click to select files or folder\n\nor\n\nDrag and drop PDF files here",
            font=ctk.CTkFont(size=14),
            text_color="gray50"
        )
        drop_label.place(relx=0.5, rely=0.5, anchor="center")
        
        self.drop_area.bind("<Button-1>", lambda e: self.select_files_or_folder())

        # Controls
        controls = ctk.CTkFrame(area_frame, fg_color="transparent")
        controls.pack(pady=20)

        self.select_btn = ctk.CTkButton(
            controls, 
            text="Select Files/Folder", 
            command=self.select_files_or_folder
        )
        self.select_btn.pack(side="left", padx=10)

        self.process_btn = ctk.CTkButton(
            controls, 
            text="Process Files", 
            fg_color="#2E8B57", 
            command=self.process_pending_files
        )
        self.process_btn.pack(side="left", padx=10)

        # Progress
        self.progress_bar = ctk.CTkProgressBar(area_frame, height=20)
        self.progress_bar.pack(fill="x", padx=40, pady=10)
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(area_frame, text="No files selected", font=ctk.CTkFont(size=12))
        self.progress_label.pack(pady=5)

        # pending paths collected via selection or drop
        self.pending_paths = []

    def setup_log_tab(self):
        log_frame = ctk.CTkFrame(self.log_tab)
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            log_frame,
            text="Activity Log",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))

        # Log text area with scrollbar
        self.log_text = ctk.CTkTextbox(
            log_frame,
            width=600,
            font=ctk.CTkFont(family="Courier", size=12)
        )
        self.log_text.pack(fill="both", expand=True, padx=15, pady=10)

        # Log controls
        log_controls = ctk.CTkFrame(log_frame, fg_color="transparent")
        log_controls.pack(fill="x", padx=15, pady=10)

        ctk.CTkButton(
            log_controls,
            text="🗑️ Clear Log",
            command=self.clear_log,
            width=100
        ).pack(side="left", padx=5)

        ctk.CTkButton(
            log_controls,
            text="📋 Copy Log",
            command=self.copy_log,
            width=100
        ).pack(side="left", padx=5)

    def show_configuration(self):
        self.tabview.set("Configuration")

    def show_process_files(self):
        self.tabview.set("Process Files")

    def change_appearance_mode(self, new_appearance_mode):
        ctk.set_appearance_mode(new_appearance_mode)

    def browse_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder_var.set(folder)

    def select_files_or_folder(self):
        # Ask user if they want to select files or a folder
        choice = messagebox.askquestion("Select Input", "Do you want to select individual files?", 
                                      detail="Click 'Yes' for files, 'No' for a folder")
        if choice == 'yes':
            files = filedialog.askopenfilenames(
                title="Select PDF files", 
                filetypes=[("PDF files", "*.pdf")]
            )
            if files:
                self.pending_paths = list(files)
                self.progress_label.configure(text=f"Selected {len(files)} files")
        else:
            folder = filedialog.askdirectory(title="Select folder containing PDFs")
            if folder:
                self.pending_paths = [folder]
                self.progress_label.configure(text=f"Selected folder: {os.path.basename(folder)}")

    def collect_pdfs_from_paths(self, paths):
        """Collect all PDF files from the selected paths"""
        pdfs = []
        for p in paths:
            if os.path.isdir(p):
                for root, dirs, files in os.walk(p):
                    for f in files:
                        if f.lower().endswith('.pdf'):
                            pdfs.append(os.path.join(root, f))
            elif os.path.isfile(p) and p.lower().endswith('.pdf'):
                pdfs.append(p)
        return pdfs

    def extract_scan_date_from_filename(self, filename):
        """Extract scan date from filename like 'Scan 2025-09-15_SCAN FILE FOR AI Testing-01.pdf'"""
        try:
            # Look for YYYY-MM-DD pattern
            date_pattern = r'(\d{4}-\d{2}-\d{2})'
            match = re.search(date_pattern, filename)
            if match:
                date_str = match.group(1)
                # Convert to YYYYMMDD format
                return datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y%m%d')
        except:
            pass
        
        # Fallback to file modification date
        try:
            return datetime.fromtimestamp(os.path.getmtime(filename)).strftime('%Y%m%d')
        except:
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
                    self.log("✅ Text extracted using PyPDF2")
                    return text
            
            # If no text found, use OCR for scanned documents
            self.log("📄 No text found, using OCR for scanned PDF...")
            return self.extract_text_with_ocr(pdf_path)
            
        except Exception as e:
            self.log(f"Error with PyPDF2 extraction, trying OCR: {str(e)}")
            return self.extract_text_with_ocr(pdf_path)

    def extract_text_with_ocr(self, pdf_path):
        """Extract text from scanned PDF using OCR"""
        if not OCR_AVAILABLE:
            self.log("❌ OCR features not available - required packages missing")
            return None

        try:
            self.log("🔍 Starting OCR processing...")
            
            # Check if tesseract is available
            try:
                pytesseract.get_tesseract_version()
            except Exception:
                self.log("❌ Tesseract OCR not found. Please install Tesseract.")
                return None

            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=200)  # Lower DPI for faster processing
            self.log(f"Converted PDF to {len(images)} images")
            
            full_text = ""
            
            for i, image in enumerate(images):
                self.log(f"Processing page {i+1}/{len(images)} with OCR...")
                
                try:
                    # Convert PIL image to OpenCV format
                    open_cv_image = np.array(image)
                    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR
                    
                    # Preprocess image for better OCR
                    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
                    
                    # Apply thresholding to get binary image
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Use pytesseract to extract text
                    page_text = pytesseract.image_to_string(thresh, config='--psm 6')
                    full_text += page_text + "\n"
                    
                    self.log(f"Page {i+1} OCR completed: {len(page_text)} characters")
                except Exception as page_error:
                    self.log(f"❌ Error processing page {i+1}: {page_error}")
                    continue
            
            if full_text.strip():
                self.log(f"✅ OCR completed successfully: {len(full_text)} total characters")
                return full_text
            else:
                self.log("❌ OCR failed to extract any text")
                return None
                
        except Exception as e:
            self.log(f"❌ OCR extraction error: {str(e)}")
            return None

    
    def extract_amount_with_regex(self, text):
        try:
            self.log("🔍 Using advanced regex to find TOTAL amount...")
            
            # Convert to lowercase for easier matching
            text_lower = text.lower()
            
            # Priority 1: Look for explicit total patterns with clear context
            total_patterns = [
                # Patterns that explicitly indicate TOTAL
                r'total\s*[\|\:\-]\s*[\$]?\s*(\d{1,3}(?:,\d{3})*\.\d{2})',  # Total | $518.78
                r'total\s+[\$]?(\d{1,3}(?:,\d{3})*\.\d{2})\s*$',  # Total $518.78 at line end
                r'amount\s+due\s*[\:\|\-]?\s*[\$]?\s*(\d{1,3}(?:,\d{3})*\.\d{2})',  # Amount Due: $518.78
                r'balance\s+due\s*[\:\|\-]?\s*[\$]?\s*(\d{1,3}(?:,\d{3})*\.\d{2})',  # Balance Due: $518.78
                r'grand\s+total\s*[\:\|\-]?\s*[\$]?\s*(\d{1,3}(?:,\d{3})*\.\d{2})',  # Grand Total: $518.78
                r'final\s+amount\s*[\:\|\-]?\s*[\$]?\s*(\d{1,3}(?:,\d{3})*\.\d{2})',  # Final Amount: $518.78
            ]
            
            for pattern in total_patterns:
                matches = re.findall(pattern, text_lower, re.MULTILINE)
                if matches:
                    amount_str = matches[-1].replace(',', '')  # Take the last match
                    try:
                        amount = float(amount_str)
                        self.log(f"✅ Found EXPLICIT TOTAL via pattern: ${amount:.2f}")
                        return amount
                    except ValueError:
                        continue
            
            # Priority 2: Look for standalone "Total" lines (common in tables)
            lines = text.split('\n')
            for i, line in enumerate(lines):
                line_clean = line.strip().lower()
                # Look for lines that are JUST the word "Total" and a dollar amount
                if re.match(r'^total\s*[\|\:\-]?\s*[\$]?\s*\d', line_clean):
                    amounts = re.findall(r'[\$]?(\d{1,3}(?:,\d{3})*\.\d{2})', line)
                    if amounts:
                        amount_str = amounts[-1].replace(',', '')
                        try:
                            amount = float(amount_str)
                            self.log(f"✅ Found TOTAL in dedicated line: ${amount:.2f}")
                            return amount
                        except ValueError:
                            continue
            
            # Priority 3: Look in the last few lines for total indicators
            last_lines = lines[-10:]  # Check last 10 lines
            for i, line in enumerate(reversed(last_lines)):
                line_clean = line.strip().lower()
                # If line contains "total" and a dollar amount
                if 'total' in line_clean and '$' in line:
                    amounts = re.findall(r'[\$](\d{1,3}(?:,\d{3})*\.\d{2})', line)
                    if amounts:
                        amount_str = amounts[-1].replace(',', '')
                        try:
                            amount = float(amount_str)
                            self.log(f"✅ Found TOTAL in bottom section: ${amount:.2f}")
                            return amount
                        except ValueError:
                            continue
            
            # Priority 4: If no clear total found, look for the largest amount that's not clearly a line item
            all_amounts = []
            # Find all dollar amounts
            amount_matches = re.findall(r'[\$](\d{1,3}(?:,\d{3})*\.\d{2})', text)
            for amt in amount_matches:
                try:
                    clean_amt = float(amt.replace(',', ''))
                    all_amounts.append(clean_amt)
                except:
                    continue
            
            if all_amounts:
                # Filter out amounts that appear to be line items
                filtered_amounts = []
                for amount in all_amounts:
                    # Check if this amount appears in a line item context
                    amount_pattern = r'\$?' + re.escape(f"{amount:.2f}") + r'(?![0-9])'
                    amount_lines = [line for line in lines if re.search(amount_pattern, line)]
                    
                    is_likely_total = True
                    for line in amount_lines:
                        line_lower = line.lower()
                        # If amount appears with quantity/price columns, it's likely a line item
                        if any(keyword in line_lower for keyword in ['quantity', 'price each', 'each', 'rate']):
                            is_likely_total = False
                            break
                        # If amount appears near "total" indicators, it's likely the total
                        if any(keyword in line_lower for keyword in ['total', 'amount due', 'balance']):
                            is_likely_total = True
                            break
                    
                    if is_likely_total:
                        filtered_amounts.append(amount)
                
                if filtered_amounts:
                    # Use the largest amount that passed filtering
                    largest_amount = max(filtered_amounts)
                    self.log(f"⚠️  Using largest filtered amount (no clear total): ${largest_amount:.2f}")
                    return largest_amount
            
            self.log("❌ No valid total amount found in document")
            return 0
            
        except Exception as e:
            self.log(f"❌ Regex amount extraction failed: {str(e)}")
            return 0
    
    def process_single_pdf(self, file_path, output_base):
        try:
            original_filename = os.path.basename(file_path)
            self.log(f"📄 Processing: {original_filename}")

            # Extract scan date from filename first
            scan_date = self.extract_scan_date_from_filename(original_filename)
            self.log(f"📅 Extracted scan date: {scan_date}")
            
            # Extract text from PDF
            text = self.extract_text_from_pdf(file_path)
            if not text:
                self.log(f"❌ Could not extract text from PDF: {original_filename}")
                return False

            self.log(f"📝 Extracted {len(text)} characters from PDF")

            # Analyze with AI - now with improved prompt for totals
            document_data = self.analyze_document_with_ai(text)
            if not document_data:
                self.log(f"❌ AI analysis failed for: {original_filename}")
                return False

            ai_amount = document_data.get('amount', 0)
            self.log(f"🤖 AI extracted amount: ${ai_amount:.2f}")

            # Always use regex as a secondary check to validate the total
            regex_amount = self.extract_amount_with_regex(text)
            
            # If AI found 0 but regex found an amount, use regex amount
            if ai_amount == 0 and regex_amount > 0:
                document_data['amount'] = regex_amount
                self.log(f"🔄 Using regex amount (AI found 0): ${regex_amount:.2f}")
            
            # If both found amounts but they're different, prefer the one that looks more like a total
            elif ai_amount > 0 and regex_amount > 0 and ai_amount != regex_amount:
                self.log(f"⚠️  Amount mismatch: AI=${ai_amount:.2f} vs Regex=${regex_amount:.2f}")
                
                # Check which one appears to be the total
                # If regex found an explicit total pattern, trust it more
                lines = text.split('\n')
                total_line_found = any('total' in line.lower() and f"${regex_amount:.2f}" in line.lower() for line in lines[-10:])
                
                if total_line_found:
                    self.log(f"🔄 Using regex amount (explicit total match): ${regex_amount:.2f}")
                    document_data['amount'] = regex_amount
                else:
                    # Otherwise, use the larger amount (likely the total)
                    larger_amount = max(ai_amount, regex_amount)
                    self.log(f"🔄 Using larger amount: ${larger_amount:.2f}")
                    document_data['amount'] = larger_amount

            # Final validation - if amount seems too small for an invoice, re-check
            final_amount = document_data.get('amount', 0)
            if 0 < final_amount < 10:  # If amount is less than $10, it's probably not the total
                self.log(f"⚠️  Amount ${final_amount:.2f} seems too small for invoice total, re-checking...")
                # Look for larger amounts in the document
                all_amounts = re.findall(r'[\$]?(\d{1,3}(?:,\d{3})*\.\d{2})', text)
                larger_amounts = [float(amt.replace(',', '')) for amt in all_amounts if float(amt.replace(',', '')) > final_amount]
                if larger_amounts:
                    new_amount = max(larger_amounts)
                    self.log(f"🔄 Using larger amount found: ${new_amount:.2f}")
                    document_data['amount'] = new_amount

            # Get vendor name for folder organization
            vendor_name = document_data.get('vendor_name', 'UnknownVendor')
            safe_vendor_name = self.safe_filename(vendor_name).replace(' ', '_')
            
            # Create vendor folder
            vendor_folder = os.path.join(output_base, safe_vendor_name)
            os.makedirs(vendor_folder, exist_ok=True)

            # Generate new filename
            new_filename = self.generate_filename(scan_date, document_data)
            if not new_filename:
                self.log(f"❌ Failed to generate filename for: {original_filename}")
                return False

            # Copy file to organized location
            target_path = os.path.join(vendor_folder, new_filename)
            
            # Handle duplicates
            base, ext = os.path.splitext(target_path)
            counter = 1
            while os.path.exists(target_path):
                target_path = f"{base}_{counter}{ext}"
                counter += 1

            # COPY the file to the new location
            shutil.copy2(file_path, target_path)
            
            # Verify the file was actually copied
            if os.path.exists(target_path):
                self.log(f"✅ SUCCESS: {original_filename} → {new_filename}")
                self.log(f"   Vendor: {vendor_name}, FINAL TOTAL: ${document_data.get('amount', 0):.2f}")
                self.log(f"   Location: {os.path.relpath(target_path, output_base)}")
                return True
            else:
                self.log(f"❌ FAILED: File was not copied to {target_path}")
                return False

        except Exception as e:
            self.log(f"❌ Error processing {os.path.basename(file_path)}: {str(e)}")
            self.log(f"Detailed error: {traceback.format_exc()}")
            return False
    
    def process_pending_files(self):
        if self.is_processing:
            messagebox.showwarning("Processing", "Files are already being processed")
            return

        if not self.pending_paths:
            messagebox.showinfo("No files", "Please select files or a folder first")
            return

        # Start processing in a separate thread to keep UI responsive
        self.is_processing = True
        self.process_btn.configure(state="disabled")
        
        def process_thread():
            try:
                # Ensure OpenAI client is initialized
                api_key = self.api_key_var.get().strip()
                if api_key and api_key.startswith('sk-'):
                    try:
                        self.client = OpenAI(api_key=api_key)
                        self.log("✅ OpenAI client initialized")
                    except Exception as e:
                        self.root.after(0, lambda: messagebox.showerror("API Error", f"Could not initialize OpenAI client: {e}"))
                        return
                else:
                    self.root.after(0, lambda: messagebox.showerror("API Key", "Please enter a valid OpenAI API key in Configuration tab"))
                    return

                # Collect all PDF files
                pdf_list = self.collect_pdfs_from_paths(self.pending_paths)
                if not pdf_list:
                    self.root.after(0, lambda: messagebox.showinfo("No PDFs", "No PDF files found in selection"))
                    return

                # Determine output folder
                output_base = self.output_folder_var.get().strip()
                if not output_base:
                    output_base = os.path.join(os.getcwd(), 'Processed_Documents')
                
                os.makedirs(output_base, exist_ok=True)

                # Process each PDF
                total = len(pdf_list)
                success_count = 0
                
                self.log(f"🚀 Starting to process {total} files...")
                
                for i, pdf_path in enumerate(pdf_list):
                    if not self.is_processing:  # Allow cancellation
                        break
                        
                    # Update progress on main thread
                    def update_progress(current=i+1, total=total, name=os.path.basename(pdf_path)):
                        self.progress_label.configure(text=f"Processing {current}/{total}: {name}")
                        self.progress_bar.set(current / total)
                    
                    self.root.after(0, update_progress)
                    
                    if self.process_single_pdf(pdf_path, output_base):
                        success_count += 1

                # Show completion message on main thread
                def show_completion():
                    self.progress_bar.set(0)
                    self.progress_label.configure(text=f"Completed: {success_count}/{total} files processed successfully")
                    self.process_btn.configure(state="normal")
                    self.is_processing = False
                    
                    messagebox.showinfo("Processing Complete", 
                                      f"Successfully processed {success_count} out of {total} files.\n\n"
                                      f"Output location: {output_base}")

                self.root.after(0, show_completion)
                
            except Exception as e:
                self.log(f"❌ Unexpected error in processing thread: {str(e)}")
                def show_error():
                    self.progress_bar.set(0)
                    self.progress_label.configure(text="Processing failed")
                    self.process_btn.configure(state="normal")
                    self.is_processing = False
                    messagebox.showerror("Processing Error", f"An error occurred: {str(e)}")
                self.root.after(0, show_error)

        # Start the processing thread
        threading.Thread(target=process_thread, daemon=True).start()

    def analyze_document_with_ai(self, text):
        if not self.client:
            self.log("❌ OpenAI client not initialized")
            return None

        # Limit text length to avoid token limits
        text = text[:12000]

        prompt = f"""
        Analyze this document text and extract the following information from invoices, bills, or financial documents:

        VENDOR NAME: Look for the company/organization that issued this document. This is typically at the top of the document as the sender.
        Examples: "The Municipal Authority of the Township of Westfall", "ABC Company Inc", "City Utilities Department"

        AMOUNT: Extract ONLY the GRAND TOTAL or FINAL AMOUNT DUE. This is the main total that should be paid.
        CRITICAL: 
        - Look for "Total", "Amount Due", "Balance Due", "Grand Total", "Final Amount"
        - IGNORE individual line items, subtotals, or partial amounts
        - IGNORE amounts that appear in quantity/price columns
        - The total is usually at the BOTTOM of the document
        - If you see multiple amounts, choose the one that appears to be the final total
        - Examples: "Total | $518.78" → 518.78, "Amount Due: $1,234.56" → 1234.56

        DOCUMENT TYPE: Identify what type of document this is (invoice, bill, receipt, statement, check, etc.)

        Return ONLY valid JSON with these exact keys: vendor_name, amount, document_type

        IMPORTANT GUIDANCE:
        - For vendor name: Look at the top of the document for the company name that sent this
        - For amount: ONLY extract the FINAL TOTAL AMOUNT, never individual line items
        - If you cannot find a clear total amount, use 0
        - If vendor name cannot be found, use "UnknownVendor"

        Document text:
        {text}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", 
                    "content": "You are an expert at extracting information from financial documents and invoices. CRITICAL: Only extract the GRAND TOTAL amount, never individual line items. Always return valid JSON format with vendor_name, amount, and document_type keys."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            result = response.choices[0].message.content.strip()
            self.log(f"🤖 AI Raw Response: {result}")
            
            # Clean the response - remove markdown code blocks if present
            result = re.sub(r'```json\s*|\s*```', '', result).strip()
            
            # Parse JSON response
            data = json.loads(result)
            
            # Validate required fields
            if 'vendor_name' not in data:
                data['vendor_name'] = 'UnknownVendor'
            if 'amount' not in data:
                data['amount'] = 0
            if 'document_type' not in data:
                data['document_type'] = 'Unknown'
                
            return data

        except json.JSONDecodeError as e:
            self.log(f"❌ Failed to parse AI response as JSON: {result}")
            # Return default data if JSON parsing fails
            return {'vendor_name': 'UnknownVendor', 'amount': 0, 'document_type': 'Unknown'}
        except Exception as e:
            self.log(f"❌ AI analysis error: {str(e)}")
            return None

    def safe_filename(self, text):
        """Convert text to safe filename"""
        if not text:
            return "Unknown"
        # Keep only alphanumeric, spaces, underscores, hyphens, and dots
        safe = re.sub(r'[^\w\s\-\.]', '', str(text))
        # Replace multiple spaces with single space
        safe = re.sub(r'\s+', ' ', safe)
        return safe.strip()

    def generate_filename(self, scan_date, document_data):
        """Generate filename in format: YYYYMMDD VendorName $Amount.pdf"""
        try:
            vendor_name = document_data.get('vendor_name', 'UnknownVendor')
            
            # Format amount
            amount = document_data.get('amount', 0)
            if isinstance(amount, (int, float)):
                amount_str = f"${amount:.2f}"
            else:
                # Extract numbers from string
                numbers = re.findall(r'\d+\.?\d*', str(amount))
                if numbers:
                    try:
                        amount_str = f"${float(numbers[0]):.2f}"
                    except:
                        amount_str = "$0.00"
                else:
                    amount_str = "$0.00"

            # Clean vendor name for filename
            safe_vendor = self.safe_filename(vendor_name)

            filename = f"{scan_date} {safe_vendor} {amount_str}.pdf"
            return filename

        except Exception as e:
            self.log(f"Error generating filename: {str(e)}")
            return None

    def test_api_key(self):
        """Test the OpenAI API key"""
        api_key = self.api_key_var.get().strip()
        if not api_key:
            messagebox.showerror("Error", "Please enter an API key")
            return

        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say 'API test successful'"}],
                max_tokens=5
            )
            messagebox.showinfo("Success", "✅ API key is valid!")
            self.status_label.configure(text="🟢 API Ready", text_color="#2E8B57")
        except Exception as e:
            error_msg = str(e)
            if "insufficient_quota" in error_msg:
                error_msg += "\n\nPlease check your billing at: https://platform.openai.com/account/billing"
            messagebox.showerror("Error", f"❌ API key test failed:\n{error_msg}")

    def save_configuration(self):
        """Save all configuration settings"""
        self.config['openai_api_key'] = self.api_key_var.get()
        self.config['output_folder'] = self.output_folder_var.get()
        
        self.save_config()

        # Reinitialize OpenAI client
        if self.config['openai_api_key']:
            try:
                self.client = OpenAI(api_key=self.config['openai_api_key'])
                self.log("✅ Configuration saved and OpenAI client initialized")
                self.status_label.configure(text="🟢 Ready", text_color="#2E8B57")
            except Exception as e:
                self.log(f"❌ Error initializing OpenAI client: {str(e)}")
                self.status_label.configure(text="🔴 API Error", text_color="#B22222")
        else:
            self.log("⚠️ Configuration saved (no API key set)")
            self.status_label.configure(text="🟡 No API Key", text_color="#FF8C00")

    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"{timestamp} - {message}\n"
        self.log_text.insert("end", log_entry)
        self.log_text.see("end")
        # Also print to console for debugging
        print(log_entry.strip())

    def clear_log(self):
        self.log_text.delete("1.0", "end")

    def copy_log(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.log_text.get("1.0", "end"))
        messagebox.showinfo("Success", "Log copied to clipboard!")

def main():
    try:
        root = ctk.CTk()
        app = PDFInvoiceRenamer(root)
        root.mainloop()
    except Exception as e:
        print(f"Failed to start application: {e}")
        print("Please make sure all dependencies are installed:")
        print("pip install customtkinter PyPDF2 openai watchdog pytesseract pdf2image opencv-python pillow")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()