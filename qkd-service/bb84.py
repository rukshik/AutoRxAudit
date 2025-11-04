"""
BB84 Quantum Key Distribution Protocol Implementation using Qiskit

This module simulates the BB84 protocol for quantum key distribution:
1. Alice prepares qubits in random bases
2. Bob measures qubits in random bases  
3. Basis reconciliation (public channel)
4. Error checking for eavesdropping detection
5. Privacy amplification to derive final key
"""

import numpy as np
import hashlib
from typing import List, Tuple, Dict
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator


class BB84Protocol:
    """
    BB84 Quantum Key Distribution Protocol
    
    Simulates quantum key exchange between Alice (sender) and Bob (receiver)
    using Qiskit quantum circuits.
    """
    
    def __init__(self, n_qubits: int = 512, target_key_bits: int = 256):
        """
        Initialize BB84 protocol
        
        Args:
            n_qubits: Number of qubits to transmit (before sifting)
            target_key_bits: Target key length after sifting and error correction
        """
        self.n_qubits = n_qubits
        self.target_key_bits = target_key_bits
        self.simulator = AerSimulator()
        
    def alice_prepare_qubits(self) -> Tuple[List[int], List[str], List[QuantumCircuit]]:
        """
        Alice prepares random qubits in random bases
        
        Returns:
            alice_bits: Random bit values [0, 1, 0, 1, ...]
            alice_bases: Random bases ['+', 'x', '+', 'x', ...]
            quantum_circuits: Prepared quantum circuits to send to Bob
        """
        # Generate random bits
        alice_bits = np.random.randint(0, 2, self.n_qubits).tolist()
        
        # Generate random bases ('+' = rectilinear, 'x' = diagonal)
        alice_bases = np.random.choice(['+', 'x'], self.n_qubits).tolist()
        
        quantum_circuits = []
        
        for bit, basis in zip(alice_bits, alice_bases):
            # Create quantum circuit
            qr = QuantumRegister(1, 'q')
            cr = ClassicalRegister(1, 'c')
            qc = QuantumCircuit(qr, cr)
            
            # Encode bit value
            if bit == 1:
                qc.x(0)  # Flip to |1⟩ state
            
            # Apply basis rotation
            if basis == 'x':
                qc.h(0)  # Hadamard for diagonal basis (|+⟩ or |-⟩)
            
            quantum_circuits.append(qc)
        
        return alice_bits, alice_bases, quantum_circuits
    
    def bob_measure_qubits(self, quantum_circuits: List[QuantumCircuit]) -> Tuple[List[int], List[str]]:
        """
        Bob measures the quantum circuits in random bases
        
        Args:
            quantum_circuits: Quantum circuits received from Alice
            
        Returns:
            bob_results: Measurement results [0, 1, 0, 1, ...]
            bob_bases: Measurement bases ['+', 'x', '+', 'x', ...]
        """
        # Generate random measurement bases
        bob_bases = np.random.choice(['+', 'x'], len(quantum_circuits)).tolist()
        
        bob_results = []
        
        for qc, basis in zip(quantum_circuits, bob_bases):
            # Create a copy of the circuit to measure
            qc_measure = qc.copy()
            
            # Apply basis rotation if measuring in diagonal basis
            if basis == 'x':
                qc_measure.h(0)
            
            # Measure
            qc_measure.measure(0, 0)
            
            # Execute on simulator
            job = self.simulator.run(qc_measure, shots=1)
            result = job.result()
            counts = result.get_counts()
            
            # Get measurement result (0 or 1)
            measured_bit = int(list(counts.keys())[0])
            bob_results.append(measured_bit)
        
        return bob_results, bob_bases
    
    def basis_reconciliation(self, 
                            alice_bases: List[str], 
                            bob_bases: List[str]) -> List[int]:
        """
        Compare bases and find matching positions (sifting)
        
        Args:
            alice_bases: Alice's preparation bases
            bob_bases: Bob's measurement bases
            
        Returns:
            matching_positions: Indices where bases matched
        """
        matching_positions = []
        for i, (a_basis, b_basis) in enumerate(zip(alice_bases, bob_bases)):
            if a_basis == b_basis:
                matching_positions.append(i)
        
        return matching_positions
    
    def sift_key(self, 
                 bits: List[int], 
                 matching_positions: List[int]) -> List[int]:
        """
        Extract sifted key from matching positions
        
        Args:
            bits: Original bit sequence
            matching_positions: Indices where bases matched
            
        Returns:
            sifted_key: Key bits where bases matched
        """
        return [bits[i] for i in matching_positions]
    
    def error_check(self, 
                   alice_sifted: List[int], 
                   bob_sifted: List[int],
                   sample_size: int = 50) -> Tuple[float, bool]:
        """
        Check for eavesdropping by comparing sample bits
        
        Args:
            alice_sifted: Alice's sifted key
            bob_sifted: Bob's sifted key
            sample_size: Number of bits to sacrifice for error checking
            
        Returns:
            error_rate: Quantum bit error rate (QBER)
            secure: True if error rate is below threshold
        """
        if len(alice_sifted) < sample_size:
            sample_size = len(alice_sifted) // 4  # Use 25% for testing
        
        # Randomly sample positions for error checking
        test_positions = np.random.choice(
            len(alice_sifted), 
            size=min(sample_size, len(alice_sifted)), 
            replace=False
        )
        
        # Count errors
        errors = 0
        for pos in test_positions:
            if alice_sifted[pos] != bob_sifted[pos]:
                errors += 1
        
        error_rate = errors / len(test_positions)
        
        # Threshold: typically 11% for BB84 (25% would indicate eavesdropping)
        secure = error_rate < 0.11
        
        return error_rate, secure
    
    def privacy_amplification(self, 
                             sifted_key: List[int], 
                             test_positions: List[int]) -> bytes:
        """
        Remove test bits and hash to final key
        
        Args:
            sifted_key: Sifted key bits
            test_positions: Positions used for error checking (to remove)
            
        Returns:
            final_key: 256-bit key as bytes
        """
        # Remove test bits
        final_bits = [bit for i, bit in enumerate(sifted_key) if i not in test_positions]
        
        # Convert bits to bytes
        bit_string = ''.join(map(str, final_bits))
        
        # Pad to ensure we have enough bits
        if len(bit_string) < 256:
            # If we don't have enough bits, pad with hash
            bit_string = bit_string + '0' * (256 - len(bit_string))
        
        # Convert to integer then to bytes
        key_int = int(bit_string[:256], 2)
        key_bytes = key_int.to_bytes(32, byteorder='big')
        
        # Hash to ensure uniform distribution and exact 256 bits
        final_key = hashlib.sha256(key_bytes).digest()
        
        return final_key
    
    def run_full_protocol(self, 
                         simulate_eavesdropping: bool = False) -> Dict:
        """
        Run complete BB84 protocol
        
        Args:
            simulate_eavesdropping: If True, inject errors to simulate Eve
            
        Returns:
            Dictionary with protocol results including keys and statistics
        """
        # Step 1: Alice prepares qubits
        alice_bits, alice_bases, quantum_circuits = self.alice_prepare_qubits()
        
        # Simulate eavesdropping by introducing errors
        if simulate_eavesdropping:
            # Eve intercepts and measures in random bases, then resends
            # This introduces ~25% error rate
            eve_bases = np.random.choice(['+', 'x'], len(quantum_circuits))
            for i, (qc, eve_basis) in enumerate(zip(quantum_circuits, eve_bases)):
                if np.random.random() < 0.25:  # 25% error injection
                    qc.x(0)  # Flip the bit
        
        # Step 2: Bob measures qubits
        bob_results, bob_bases = self.bob_measure_qubits(quantum_circuits)
        
        # Step 3: Basis reconciliation (public channel)
        matching_positions = self.basis_reconciliation(alice_bases, bob_bases)
        
        # Step 4: Sift keys
        alice_sifted = self.sift_key(alice_bits, matching_positions)
        bob_sifted = self.sift_key(bob_results, matching_positions)
        
        # Step 5: Error checking
        error_rate, secure = self.error_check(alice_sifted, bob_sifted)
        
        if not secure:
            return {
                'success': False,
                'error_rate': error_rate,
                'secure': False,
                'message': 'Eavesdropper detected! Error rate too high.',
                'alice_key': None,
                'bob_key': None
            }
        
        # Step 6: Privacy amplification
        sample_size = min(50, len(alice_sifted) // 4)
        test_positions = np.random.choice(
            len(alice_sifted), 
            size=sample_size, 
            replace=False
        ).tolist()
        
        alice_key = self.privacy_amplification(alice_sifted, test_positions)
        bob_key = self.privacy_amplification(bob_sifted, test_positions)
        
        # Verify keys match
        keys_match = alice_key == bob_key
        
        return {
            'success': keys_match and secure,
            'alice_key': alice_key.hex() if keys_match else None,
            'bob_key': bob_key.hex() if keys_match else None,
            'error_rate': error_rate,
            'secure': secure,
            'keys_match': keys_match,
            'initial_qubits': self.n_qubits,
            'sifted_bits': len(alice_sifted),
            'final_key_bits': 256,
            'sifting_efficiency': len(matching_positions) / self.n_qubits,
            'message': 'Key exchange successful!' if keys_match and secure else 'Key exchange failed!'
        }


def simulate_qkd_exchange(n_qubits: int = 512, 
                         target_bits: int = 256,
                         simulate_eve: bool = False) -> Dict:
    """
    Convenience function to run BB84 protocol
    
    Args:
        n_qubits: Number of qubits to exchange
        target_bits: Target key length
        simulate_eve: Whether to simulate eavesdropping
        
    Returns:
        Protocol results dictionary
    """
    bb84 = BB84Protocol(n_qubits=n_qubits, target_key_bits=target_bits)
    return bb84.run_full_protocol(simulate_eavesdropping=simulate_eve)


if __name__ == "__main__":
    # Test the protocol
    print("Running BB84 Protocol Test...")
    print("=" * 60)
    
    # Normal execution
    print("\n1. Normal key exchange (no eavesdropper):")
    result = simulate_qkd_exchange()
    print(f"   Success: {result['success']}")
    print(f"   Error Rate: {result['error_rate']:.4f}")
    print(f"   Sifted Bits: {result['sifted_bits']}/{result['initial_qubits']}")
    print(f"   Key (Alice): {result['alice_key'][:32]}...")
    print(f"   Key (Bob):   {result['bob_key'][:32]}...")
    print(f"   Keys Match: {result['keys_match']}")
    
    # With eavesdropper
    print("\n2. Key exchange with eavesdropper:")
    result_eve = simulate_qkd_exchange(simulate_eve=True)
    print(f"   Success: {result_eve['success']}")
    print(f"   Error Rate: {result_eve['error_rate']:.4f}")
    print(f"   Secure: {result_eve['secure']}")
    print(f"   Message: {result_eve['message']}")
