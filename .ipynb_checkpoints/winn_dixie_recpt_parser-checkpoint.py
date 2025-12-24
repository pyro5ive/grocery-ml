import re
from typing import List, Dict, Any, Optional
import unicodedata

class WinnDixieRecptParser:

    STORE_MANAGER_PATTERN = re.compile(r'store\s*manager\s*:\s*(?P<manager>.+)', re.IGNORECASE)
    CASHIER_PATTERN       = re.compile(r'your\s*cashier\s*:\s*(?P<cashier>.+)', re.IGNORECASE)

    # Relaxed header now matches more OCR variations
    HEADER_PATTERN = re.compile(r'Reg\s+You\s*Pay', re.IGNORECASE)

    STOP_PATTERN = re.compile(r'\btotal\s*items\s*sold\b', re.IGNORECASE)
    COUNT_PATTERN = re.compile(
        r'\btotal\s*(?:number\s*of\s*)?items\s*sold\b[^0-9]*([0-9]+)',
        re.IGNORECASE
    )

    DATETIME_PATTERN = re.compile(
        r'(?P<date>\d{1,2}/\d{1,2}/\d{2,4})\s*(?:,?\s*at\s+)?(?P<time>\d{1,2}:\d{2}(?:\s*[AP]M)?)',
        re.IGNORECASE
    )

    ITEM_PATTERN = re.compile(
        r'^(?:QTY\s*(?P<qty>\d+)\s+)?'
        r'(?P<item>.*?)(?=\s+\$)'
        r'\s+\$(?P<reg>\d+\.\d{2})'
        r'\s+\$(?P<pay>\d+\.\d{2})'
        r'(?:\s+[A-Z])?\s*$',
        re.IGNORECASE
    )

    #########################################################################
    def parse(self, text: str) -> Dict[str, Any]:
        # Normalize all lines
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
                'validation': {
                    'rowsMatchReported': False,
                    'qtyMatchReported': False,
                    'rowsCount': 0,
                    'qtySum': 0
                }
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
            # skip lines
            if self.clean_you_save(ln) or self.clean_coupon(ln):
                continue
            if not ln.strip():
                continue

            # stop condition
            m_stop = self.COUNT_PATTERN.search(ln)
            if m_stop:
                reported = int(m_stop.group(1))
                break

            # try to match an item line
            m = self.ITEM_PATTERN.match(ln)
            if not m:
                continue

            qty = int(m.group('qty')) if m.group('qty') else 1
            item_name = m.group("item")

            # Fix OCR prefixes like "qty-3-love-cheese"
            item_name, qty = self.fix_ocr_qty_prefix(item_name, qty)

            # Normalize name
            item_name = self.clean_item_name(item_name)

            items.append({
                "item": item_name,
                "qty": qty,
                "reg": float(m.group("reg")),
                "youPay": float(m.group("pay")),
            })

        return items, reported

    #########################################################################
    def fix_ocr_qty_prefix(self, item_name: str, qty: int):
        """
        Detect OCR mistakes:
            qty-3-sprite
            qty 2 eggs
            oty-4-coke
        → qty becomes the real quantity
        → prefix removed from item name
        """
        m = re.match(r'^(?:q|o)ty[-\s]*(\d+)\s*[-]?\s*(.+)$', item_name, re.IGNORECASE)
        if m:
            correct_qty = int(m.group(1))
            clean_name = m.group(2).strip()
            return clean_name, correct_qty
        return item_name, qty      
    #########################################################################
    def clean_item_name(self, item: str) -> str:
        if item is None:
            return ""

        item = str(item).strip()

        # "&" → "and"
        item = item.replace("&", "and")

        # whitespace → hyphen
        item = re.sub(r"\s+", "-", item)

        # remove invalid chars
        item = re.sub(r"[^A-Za-z0-9-]", "", item)

        # collapse multiple hyphens
        item = re.sub(r"-{2,}", "-", item)

        return item.lower()

    #########################################################################
    def clean_coupon(self, line: str) -> bool:
        return re.search(r'^\s*CPN\s+SCANNED\s+COUPON', line, re.IGNORECASE) is not None

    #########################################################################
    def clean_you_save(self, line: str) -> bool:
        return re.search(r'^\s*you\s*save\b', line, re.IGNORECASE) is not None

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
    def validate_counts(self, items: List[Dict[str, Any]], reported: Optional[int]):
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
    
    @staticmethod
    def remove_duplicate_receipt_files(df):
        """
        Remove whole source files that contain an identical receipt
        to another file with the same date+time.
        Minimal console output. Resets index at end.
        """

        df["__signature"] = (
            df["date"].astype(str) + "|" +
            df["time"].astype(str) + "|" +
            df["item"].astype(str) + "|" 
            #df["qty"].astype(str) + "|" +
            #df["youPay"].astype(str) + "|" +
            #df["reg"].astype(str) + "|" +
            #df["reportedItemsSold"].astype(str) + "|" +
            #df["cashier"].astype(str) + "|" +
            #df["manager"].astype(str)
        )
    
        keep_sources = set()
    
        for (dt_date, dt_time), group in df.groupby(["date", "time"]):
    
            # Build signature per source
            source_signatures = {}
            for source, rows in group.groupby("source"):
                sig = tuple(sorted(rows["__signature"].tolist()))
                source_signatures[source] = sig
    
            # signature → list of sources
            signature_groups = {}
            for src, sig in source_signatures.items():
                signature_groups.setdefault(sig, []).append(src)
    
            # Handle duplicates
            for sig, sources in signature_groups.items():
                if len(sources) == 1:
                    keep_sources.add(sources[0])
                    continue
    
                sorted_sources = sorted(sources)
                kept = sorted_sources[0]
                removed = sorted_sources[1:]
    
                # Minimal output
                print(f"DUP: {dt_date} {dt_time} → keep {kept} ← drop {', '.join(removed)}")
    
                keep_sources.add(kept)
    
        # Filter and clean
        result = df[df["source"].isin(keep_sources)].copy()
        result.drop(columns=["__signature"], inplace=True)
    
        # ✔ Reset index here
        result.reset_index(drop=True, inplace=True)
    
        return result
#################################################################

