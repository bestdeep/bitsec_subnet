# Arithmetic Overflow and Underflow Vulnerability
Integer overflow/underflow in smart contracts is a critical vulnerability that occurs when arithmetic operations exceed the maximum or minimum limits of the data type. This is particularly dangerous in versions of Solidity < 0.8 because:

- Integers in Solidity have fixed sizes (uint8 = 8 bits, uint256 = 256 bits, etc.)
- When an operation causes the value to exceed the maximum (2^n - 1) or go below the minimum (0), it will wrap around without any error (versions >= 0.8 throw an error instead)
- This wrapping behavior can be exploited to bypass time locks, manipulate balances, or break other contract invariants

For example, in a uint8 (8 bits):
- Maximum value: 255 (2^8 - 1)
- If you add 1 to 255 → it overflows to 0
- If you subtract 1 from 0 → it underflows to 255

## Why is this dangerous?
In the example contract, the vulnerability allows an attacker to:
1. Deposit funds with a normal 1-week timelock
2. Exploit integer overflow to manipulate the lockTime
3. Withdraw funds immediately, bypassing the timelock entirely
The attack works by finding a value X that when added to the current lockTime T will cause an overflow back to 0:
