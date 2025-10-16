import numpy as np

class HRPRSCoder:
    """RS encoding/decoding for HRP PHY
    
    Performs Reed-Solomon encoding or decoding as per Sec. 15.3.3.2 
    in IEEE Std 802.15.4™‐2020.
    """
    
    def __init__(self):
        # RS parameters as per IEEE 802.15.4-2020
        self.N = 63  # Codeword length
        self.K = 55  # Message length
        self.M = 6   # Symbol size in bits
        
        # Generator polynomial coefficients (GF(2^6))
        # genPoly = 'x8 + 55x7 + 61x6 + 37x5 + 48x4 + 47x3 + 20x2 + 6x1 + 22'
        # These are the coefficients in descending order
        self.gen_poly_coefs = [1, 55, 61, 37, 48, 47, 20, 6, 22]
        
        # Primitive polynomial: 1 + x + x^6 = x^6 + x + 1
        self.prim_poly = 0b1000011  # Binary: 1000011
        
        # Initialize GF(2^6) tables
        self._init_galois_field()
        self._init_generator_polynomial()
        
    def _init_galois_field(self):
        """Initialize Galois Field GF(2^6) lookup tables"""
        n = 2**self.M
        self.gf_exp = np.zeros(n * 2, dtype=np.int32)
        self.gf_log = np.zeros(n, dtype=np.int32)
        
        x = 1
        for i in range(n - 1):
            self.gf_exp[i] = x
            self.gf_log[x] = i
            x <<= 1
            if x & n:
                x ^= self.prim_poly
        
        # Extend exp table for convenience
        for i in range(n - 1, 2 * n - 2):
            self.gf_exp[i] = self.gf_exp[i - (n - 1)]
    
    def _gf_mul(self, x: int, y: int) -> int:
        """Galois Field multiplication"""
        if x == 0 or y == 0:
            return 0
        return self.gf_exp[self.gf_log[x] + self.gf_log[y]]
    
    def _gf_div(self, x: int, y: int) -> int:
        """Galois Field division"""
        if y == 0:
            raise ZeroDivisionError("Division by zero in GF")
        if x == 0:
            return 0
        return self.gf_exp[(self.gf_log[x] - self.gf_log[y]) % 63]
    
    def _init_generator_polynomial(self):
        """Initialize generator polynomial for RS code
        
        The generator polynomial is directly given in the specification.
        genPoly = 'x^8 + 55x^7 + 61x^6 + 37x^5 + 48x^4 + 47x^3 + 20x^2 + 6x + 22'
        
        This means the generator has 9 coefficients (degree 8 = N-K = 63-55)
        """
        # The generator polynomial coefficients are given directly
        # in descending order of powers
        self.generator = np.array(self.gen_poly_coefs, dtype=np.int32)
    
    def _rs_encode_block(self, msg: np.ndarray) -> np.ndarray:
        """Encode a single block using RS encoding
        
        Systematic encoding: codeword = [message | parity]
        Parity is computed as the remainder of (message * x^(N-K)) / generator
        """
        # msg should be K symbols (55 symbols)
        msg = np.array(msg, dtype=np.int32)
        
        # Initialize the shift register with zeros
        # We need N-K = 8 parity symbols
        n_parity = self.N - self.K
        parity = np.zeros(n_parity, dtype=np.int32)
        
        # Systematic encoding using synthetic division
        # Process each message symbol
        for i in range(len(msg)):
            # Feedback term
            feedback = msg[i] ^ parity[0]
            
            # Shift parity symbols
            for j in range(n_parity - 1):
                # parity[j] = parity[j+1] ^ (feedback * generator[j+1])
                if feedback != 0:
                    parity[j] = parity[j + 1] ^ self._gf_mul(feedback, self.generator[j + 1])
                else:
                    parity[j] = parity[j + 1]
            
            # Last parity symbol
            if feedback != 0:
                parity[n_parity - 1] = self._gf_mul(feedback, self.generator[n_parity])
            else:
                parity[n_parity - 1] = 0
        
        # Codeword = message + parity
        return np.concatenate([msg, parity])
    
    def _rs_decode_block(self, received: np.ndarray) -> np.ndarray:
        """Decode a single block using RS decoding"""
        # Compute syndromes
        syndromes = self._compute_syndromes(received)
        
        # If all syndromes are zero, no errors
        if np.all(syndromes == 0):
            return received[:self.K]
        
        # Attempt error correction using Berlekamp-Massey algorithm
        try:
            corrected = self._correct_errors(received, syndromes)
            return corrected[:self.K]
        except:
            # If correction fails, return uncorrected message
            return received[:self.K]
    
    def _compute_syndromes(self, received: np.ndarray) -> np.ndarray:
        """Compute syndrome polynomials
        
        Syndrome S_i = sum(c_j * alpha^(i*j)) for i = 1 to N-K
        where alpha is the primitive element
        """
        n_syndromes = self.N - self.K
        syndromes = np.zeros(n_syndromes, dtype=np.int32)
        
        for i in range(n_syndromes):
            syndrome = 0
            for j in range(len(received)):
                if received[j] != 0:
                    # alpha^((i+1)*j)
                    power = ((i + 1) * j) % 63
                    syndrome ^= self._gf_mul(received[j], self.gf_exp[power])
            syndromes[i] = syndrome
        
        return syndromes
    
    def _correct_errors(self, received: np.ndarray, syndromes: np.ndarray):
        """Berlekamp-Massey algorithm for error correction"""
        # Simplified error correction
        # For full implementation, use Berlekamp-Massey + Chien search
        
        # For now, just return the received codeword
        # A complete implementation would:
        # 1. Use Berlekamp-Massey to find error locator polynomial
        # 2. Use Chien search to find error locations
        # 3. Use Forney algorithm to find error values
        # 4. Correct the errors
        
        return received
    
    def _bit2int(self, bits: np.ndarray, m: int, msb_first: bool = False) -> np.ndarray:
        """Convert bits to integers (symbols)
        
        Args:
            bits: Array of bits
            m: Number of bits per symbol
            msb_first: If False, use right-msb (LSB first)
        """
        n_symbols = len(bits) // m
        symbols = np.zeros(n_symbols, dtype=np.int32)
        
        for i in range(n_symbols):
            symbol_bits = bits[i*m:(i+1)*m]
            if msb_first:
                # MSB first (left-msb)
                symbol = 0
                for bit in symbol_bits:
                    symbol = (symbol << 1) | int(bit)
            else:
                # LSB first (right-msb)
                symbol = 0
                for j, bit in enumerate(symbol_bits):
                    symbol |= int(bit) << j
            symbols[i] = symbol
        
        return symbols
    
    def _int2bit(self, symbols: np.ndarray, m: int, msb_first: bool = False) -> np.ndarray:
        """Convert integers (symbols) to bits
        
        Args:
            symbols: Array of symbols
            m: Number of bits per symbol
            msb_first: If False, use right-msb (LSB first)
        """
        n_bits = len(symbols) * m
        bits = np.zeros(n_bits, dtype=np.int32)
        
        for i, symbol in enumerate(symbols):
            if msb_first:
                # MSB first (left-msb)
                for j in range(m):
                    bits[i*m + j] = (symbol >> (m - 1 - j)) & 1
            else:
                # LSB first (right-msb)
                for j in range(m):
                    bits[i*m + j] = (symbol >> j) & 1
        
        return bits


def hrpRS(input_bits: np.ndarray, do_encode: bool) -> np.ndarray:
    """RS encoding/decoding for HRP PHY
    
    Args:
        input_bits: Binary column vector of any length
        do_encode: True for encoding, False for decoding
    
    Returns:
        Encoded or decoded output
    """
    
    # Initialize coder (singleton pattern)
    if not hasattr(hrpRS, 'coder'):
        hrpRS.coder = HRPRSCoder()
    
    coder = hrpRS.coder
    M = coder.M
    
    # Ensure input is column vector
    input_bits = np.asarray(input_bits).flatten()
    
    block_size = 330  # 330 for encoding
    if not do_encode:
        block_size = 330 + M * (63 - 55)  # 378 for decoding
    
    output = np.array([], dtype=np.int32)
    
    # Process PSDU in blocks
    n_blocks = int(np.ceil(len(input_bits) / block_size))
    
    for block_idx in range(n_blocks):
        # Extract this block
        start_idx = block_size * block_idx
        end_idx = min(len(input_bits), block_size * (block_idx + 1))
        this_block = input_bits[start_idx:end_idx]
        
        I = len(this_block)
        
        # a) Addition of dummy bits
        n_dummy = block_size - I
        in_padded = np.concatenate([np.zeros(n_dummy, dtype=np.int32), this_block])
        
        # b) Bit-to-symbol conversion, with right-msb
        msb_first = False
        in_padded_int = coder._bit2int(in_padded, M, msb_first)
        
        # c) Encoding/Decoding
        if do_encode:
            tmp = coder._rs_encode_block(in_padded_int[:block_size // M])
        else:
            tmp = coder._rs_decode_block(in_padded_int[:block_size // M])
        # d) Symbol to bit conversion
        out_padded_bits = coder._int2bit(tmp, M, msb_first)
        
        # e) Removal of dummy bits, concatenation with previous outputs
        removal_start = max(0, block_size - I)
        out_block = out_padded_bits[removal_start:]
        output = np.concatenate([output, out_block])
    
    return output


# Test and debug function
def test_encoding():
    """Test function to compare with MATLAB results"""
    
    # Create test input - you can replace this with your MATLAB test data
    test_input = np.array([1,0,1,0,1,0,1,0 ,0,1,0,1,0,1,0,1, 1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 0,1,0,1,0,1,0,1, 1,0,1,0,1,0,1,0])  # 55 symbols worth
    
    coder = HRPRSCoder()
    
    # Test GF operations
    print("Testing GF(2^6) operations:")
    print(f"GF exp[0] = {coder.gf_exp[0]}")
    print(f"GF exp[1] = {coder.gf_exp[1]}")
    print(f"GF exp[5] = {coder.gf_exp[5]}")
    
    # Test bit to symbol conversion
    symbols = coder._bit2int(test_input, 6, msb_first=False)
    print(f"\nFirst 5 symbols: {symbols[:5]}")
    
    # Test encoding
    encoded = coder._rs_encode_block(symbols)
    print(f"\nEncoded length: {len(encoded)} (should be 63)")
    print(f"First 5 encoded symbols: {encoded[:5]}")
    print(f"Last 8 parity symbols: {encoded[55:]}")
    
    return coder, symbols, encoded


if __name__ == "__main__":
        # Test encoding
    # input_data = np.random.randint(0, 2, 330)  # 330 bits
    input_data = np.array([1,0,1,0,1,0,1,0 ,0,1,0,1,0,1,0,1, 1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 0,1,0,1,0,1,0,1, 1,0,1,0,1,0,1,0])
    encoded = hrpRS(input_data, do_encode=True)
    print(f"Input length: {len(input_data)}")
    print(f"Encoded length: {len(encoded)} {encoded}")
    
    # Test decoding
    decoded = hrpRS(encoded, do_encode=False)
    print(f"Decoded length: {len(decoded)}")
    print(f"Match: {np.array_equal(input_data, decoded)}")