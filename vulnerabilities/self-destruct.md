# Self Destruct
The `selfdestruct` operation in Solidity is a special function that allows a contract to be deleted from the blockchain. This creates a significant vulnerability because:
- It forcibly sends all contract balance to a designated address
- The receiving contract cannot reject these funds
- This forced transfer can break contract invariants that rely on balance checks
- Once a contract is self-destructed, all its code and storage is removed from the blockchain
