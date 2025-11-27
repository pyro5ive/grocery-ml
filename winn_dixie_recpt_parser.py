import re
from typing import List, Dict, Any, Optional, Tuple
import unicodedata

class WinnDixieRecptParser:

    STORE_MANAGER_PATTERN = re.compile(r'store\s*manager\s*:\s*(?P<manager>.+)', re.IGNORECASE)
    CASHIER_PATTERN       = re.compile(r'your\s*cashier\s*:\s*(?P<cashier>.+)', re.IGNORECASE)
    HEADER_PATTERN = re.compile(r'^\s*Reg\s+You\s*Pay\s*$', re.IGNORECASE)
    STOP_PATTERN     = re.compile(r'\btotal\s*items\s*sold\b', re.IGNORECASE)
    COUNT_PATTERN = re.compile( r'\btotal\s*(?:number\s*of\s*)?items\s*sold\b[^0-9]*([0-9]+)',re.IGNORECASE)
    
    DATETIME_PATTERN = re.compile( r'(?P<date>\d{1,2}/\d{1,2}/\d{2,4})\s*(?:,?\s*at\s+)?(?P<time>\d{1,2}:\d{2}(?:\s*[AP]M)?)',re.IGNORECASE)

    ITEM_PATTERN = re.compile(
        r'^(?:QTY\s*(?P<qty>\d+)\s+)?'      # optional QTY
        r'(?P<item>.*?)(?=\s+\$)'           # item text up to first $
        r'\s+\$(?P<reg>\d+\.\d{2})'         # Reg price
        r'\s+\$(?P<pay>\d+\.\d{2})'         # You Pay price
        r'(?:\s+[A-Z])?\s*$',               # optional trailing flag like B
        re.IGNORECASE
    )


    #ef parse_file_to_rows(path: Path) -> List[Dict[str, Any]]:
    #    text = path.read_text(encoding='utf-8', errors='ignore')
    #    #print(text)
    #    result = parse_receipt_text(text)
    #    print(result)
    #    rows: List[Dict[str, Any]] = []
    #    for r in result['items']:
    #        rows.append({
    #            'source': path.name,
    #            'date': result['date'],
    #            'time': result['time'],
    #            'item': r['item'],
    #            'qty': r['qty'],
    #            'reg': r['reg'],
    #            'youPay': r['youPay'],
    #            'reportedItemsSold': result['reported'],
    #            'rowsMatchReported': result['validation']['rowsMatchReported'],
    #            'qtyMatchReported': result['validation']['qtyMatchReported']
    #       })
    #   return rows
    #########################################################################

    def parse(self, text: str) -> Dict[str, Any]:
        lines = [self.normalize_spaces(ln) for ln in text.splitlines() if ln.strip()]
    
        date, time = self.extract_datetime(lines)
        manager = self.extract_store_manager(lines)
        cashier = self.extract_cashier(lines)
    
        start = self.find_header_index(lines)
        if start is None:
            return {
                'items': [],
                'reported': None,
                'date': date,
                'time': time,
                'manager': manager,
                'cashier': cashier,
                'validation': {'rowsMatchReported': False, 'qtyMatchReported': False, 'rowsCount': 0, 'qtySum': 0}
            }

        items, reported = self.parse_items(lines, start)
        validation = self.validate_counts(items, reported)

        return {
            'items': items,
            'reported': reported,
            'date': date,
            'time': time,
            'manager': manager,
            'cashier': cashier,
            'validation': validation
        }

    #########################################################################

    def parse_items(self, lines, start_idx):
        items = []
        reported = None
        for ln in lines[start_idx:]:
            if self.clean_you_save(ln):
                continue
            if self.clean_you_save(ln) or self.clean_coupon(ln):
                continue    
            if not ln.strip():
                continue
    
            m_stop = self.COUNT_PATTERN.search(ln)
            if m_stop:
                reported = int(m_stop.group(1))
                break
    
            m = self.ITEM_PATTERN.match(ln)
            if m:
                qty = int(m.group('qty')) if m.group('qty') else 1
                items.append({
                    "item": m.group("item"),
                    "qty": qty,
                    "reg": float(m.group("reg")),
                    "youPay": float(m.group("pay")),
                })
        return items, reported
    
    #########################################################################
    
    def clean_coupon(self, line: str) -> bool:
        """Skip scanned coupon lines."""
        return re.search(r'^\s*CPN\s+SCANNED\s+COUPON', line, re.IGNORECASE) is not None
    #########################################################################
    
    def clean_you_save(self, line: str) -> bool:
        """Return True if line should be skipped because it starts with 'You save'."""
        return re.search(r'^\s*you\s*save\b', line, re.IGNORECASE) is not None
    #########################################################################
    
    #def clean_lines(text: str, exclude: Optional[List[re.Pattern]] = None) -> List[str]:
    #    if exclude is None:
    #        exclude = [
    #            re.compile(r'\b(subtotal|tax|change|thank\s*you|cash|card|tender)\b', re.IGNORECASE),
    #            re.compile(r'^\s*$')
    #        ]
    #    return [ln.strip() for ln in text.splitlines() if not any(p.search(ln) for p in exclude)]
    #########################################################################
    
    def normalize_spaces(self, s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        return re.sub(r'\s+', ' ', s.strip())
    #########################################################################
    
    def extract_datetime(self, lines):
        for ln in lines:
            m = self.DATETIME_PATTERN.search(ln)
            if m:
                return m.group('date'), m.group('time')
        return None, None
    #########################################################################
        
    def find_header_index(self, lines: List[str]) -> Optional[int]:
        for i, ln in enumerate(lines):
            if self.HEADER_PATTERN.search(ln):
                return i + 1
        return None
    #########################################################################
    
    def validate_counts(self, items: List[Dict[str, Any]], reported: Optional[int]) -> Dict[str, Any]:
        rows_count = len(items)
        qty_sum = sum(r['qty'] for r in items)
        return {
            'rowsMatchReported': (reported is not None and rows_count == reported),
            'qtyMatchReported': (reported is not None and qty_sum == reported),
            'rowsCount': rows_count,
            'qtySum': qty_sum
        }
    #########################################################################
    
    def extract_store_manager(self, lines: List[str]) -> Optional[str]:
        for ln in lines:
            m = self.STORE_MANAGER_PATTERN.search(ln)
            if m:
                return m.group('manager').strip()
        return None
    #########################################################################
    
    def extract_cashier(self, lines: List[str]) -> Optional[str]:
        for ln in lines:
            m = self.CASHIER_PATTERN.search(ln)
            if m:
                return m.group('cashier').strip()
        return None
     #########################################################################
