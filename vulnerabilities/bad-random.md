# Bad Randomness Vulnerability
- The blockchain does not provide any cryptographically secure source of randomness. Block hashes in isolation are cryptographically random, however, a malicious miner can modify block headers, introduce additional transactions, and choose not to publish blocks in order to influence the resulting hashes. Therefore, miner-influenced values like block hashes and timestamps should never be used as a source of randomness.
- Everything in a contract is publicly visible. Random numbers cannot be generated or stored in the contract until after all lottery entries have been stored.
- Computers will always be faster than the blockchain. Any number that the contract could generate can potentially be precalculated off-chain before the end of the block.
