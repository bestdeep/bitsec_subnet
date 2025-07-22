import pytest
from unittest.mock import AsyncMock, MagicMock
import numpy as np
import json

from neurons.validator_proxy import ValidatorProxy
from bitsec.protocol import PredictionResponse, Vulnerability, VulnerabilityByMiner, LineRange
from bitsec.base.vulnerability_category import VulnerabilityCategory

@pytest.fixture
def mock_validator() -> MagicMock:
    """
    Create a mock validator with necessary attributes.
    
    Returns:
        MagicMock: A mock validator object.
    """
    validator = MagicMock()
    validator.wallet = MagicMock()
    validator.metagraph = MagicMock()
    validator.metagraph.uids = np.array([0, 1, 2])
    validator.metagraph.R = np.array([0.1, 0.2, 0.3])
    validator.metagraph.I = np.array([0.4, 0.5, 0.6])
    validator.metagraph.E = np.array([0.7, 0.8, 0.9])
    validator.metagraph.axons = {0: MagicMock(), 1: MagicMock(), 2: MagicMock()}
    validator.last_responding_miner_uids = [1, 2]
    validator.config = MagicMock()
    validator.config.neuron.full_path = "/tmp"
    validator.config.neuron.sample_size = 2
    validator.config.proxy.port = None
    return validator

@pytest.fixture
def mock_request() -> MagicMock:
    """
    Create a mock request object.
    
    Returns:
        MagicMock: A mock request object.
    """
    request = MagicMock()
    request.json = AsyncMock(return_value={"code": "contract Test {}"})
    return request

@pytest.fixture
def mock_prediction_responses() -> list[PredictionResponse]:
    """
    Create mock prediction responses.
    
    Returns:
        list[PredictionResponse]: A list of mock prediction responses.
    """
    vuln1 = Vulnerability(
        category=VulnerabilityCategory.ARITHMETIC_OVERFLOW_AND_UNDERFLOW,
        title="Arithmetic overflow",
        severity="50_medium",
        line_ranges=[LineRange(start=1, end=9)],
        description="Can lead to loss of funds",
        vulnerable_code="",
        code_to_exploit="",
        rewritten_code_to_fix_vulnerability="",
    )

    vuln2 = Vulnerability(
        category=VulnerabilityCategory.WEAK_ACCESS_CONTROL,
        title="Sensitive Data Exposure via Weak Access Control",
        severity="85_high",
        line_ranges=[LineRange(start=10, end=20)],
        description="Allows unauthorized access to sensitive data",
        vulnerable_code="",
        code_to_exploit="",
        rewritten_code_to_fix_vulnerability=""
    )

    vuln3 = Vulnerability(
        category=VulnerabilityCategory.REENTRANCY,
        title="Reentrancy Attack Risk",
        severity="99_critical",
        line_ranges=[LineRange(start=21, end=30)],
        description="Can lead to loss of funds",
        vulnerable_code="",
        code_to_exploit="",
        rewritten_code_to_fix_vulnerability=""
    )

    vuln4 = Vulnerability(
        category=VulnerabilityCategory.INCORRECT_CALCULATION,
        title="Contract Termination via Logic Error",
        severity="85_high",
        line_ranges=[LineRange(start=30, end=40)],
        description="Allows unauthorized access to terminate the contract",
        vulnerable_code="",
        code_to_exploit="",
        rewritten_code_to_fix_vulnerability=""
    )

    vuln5 = Vulnerability(
        category=VulnerabilityCategory.BAD_RANDOMNESS,
        title="Predictable Randomness Allows Exploitation",
        severity="50_medium",
        line_ranges=[LineRange(start=30, end=40)],
        description="Allows unauthorized access to terminate the contract",
        vulnerable_code="",
        code_to_exploit="",
        rewritten_code_to_fix_vulnerability=""
    )

    vuln6 = Vulnerability(
        category=VulnerabilityCategory.FRONT_RUNNING,
        title="Susceptibility to Front-Running",
        severity="85_high",
        line_ranges=[LineRange(start=30, end=40)],
        description="Allows unauthorized access to terminate the contract",
        vulnerable_code="",
        code_to_exploit="",
        rewritten_code_to_fix_vulnerability=""
    )

    
    return [
        PredictionResponse(prediction=True, vulnerabilities=[vuln1, vuln2, vuln3]),
        PredictionResponse(prediction=True, vulnerabilities=[vuln4, vuln5, vuln6])
    ]

@pytest.fixture
def sample_vulnerability() -> Vulnerability:
    """
    Create a sample vulnerability for testing.
    
    Returns:
        Vulnerability: A sample vulnerability instance
    """
    return Vulnerability(
        title="Arithmetic overflow",
        severity="50_medium",
        category=VulnerabilityCategory.ARITHMETIC_OVERFLOW_AND_UNDERFLOW,
        line_ranges=[LineRange(start=1, end=9)],
        description="Can lead to arithmetic overflow",
        vulnerable_code="function add(uint a, uint b) returns (uint) { return a + b; }",
        code_to_exploit="add(MAX_UINT, 1)",
        rewritten_code_to_fix_vulnerability="function add(uint a, uint b) returns (uint) { uint c = a + b; require(c >= a); return c; }",
    )

def test_vulnerability_by_miner_from_tuple(sample_vulnerability: Vulnerability) -> None:
    """
    Test creating VulnerabilityByMiner from tuple.
    
    Args:
        sample_vulnerability: Sample vulnerability fixture
    """
    miner_id = "123"
    vuln_by_miner = VulnerabilityByMiner.from_tuple((miner_id, sample_vulnerability))
    
    assert vuln_by_miner.miner_id == miner_id
    assert vuln_by_miner.category == sample_vulnerability.category
    assert vuln_by_miner.line_ranges == sample_vulnerability.line_ranges
    assert vuln_by_miner.description == sample_vulnerability.description
    assert vuln_by_miner.vulnerable_code == sample_vulnerability.vulnerable_code
    assert vuln_by_miner.code_to_exploit == sample_vulnerability.code_to_exploit
    assert vuln_by_miner.rewritten_code_to_fix_vulnerability == sample_vulnerability.rewritten_code_to_fix_vulnerability

def test_vulnerability_by_miner_to_tuple(sample_vulnerability: Vulnerability) -> None:
    """
    Test converting VulnerabilityByMiner to tuple.
    
    Args:
        sample_vulnerability: Sample vulnerability fixture
    """
    miner_id = "123"
    vuln_by_miner = VulnerabilityByMiner(
        miner_id=miner_id,
        **sample_vulnerability.model_dump()
    )
    
    result_miner_id, result_vuln = vuln_by_miner.to_tuple()
    
    assert result_miner_id == miner_id
    assert isinstance(result_vuln, Vulnerability)
    assert result_vuln.model_dump() == sample_vulnerability.model_dump()

def test_vulnerability_by_miner_from_json(sample_vulnerability: Vulnerability) -> None:
    """
    Test creating VulnerabilityByMiner from JSON.
    
    Args:
        sample_vulnerability: Sample vulnerability fixture
    """
    miner_id = "123"
    data = {
        "miner_id": miner_id,
        **sample_vulnerability.model_dump()
    }
    
    # Test with dict
    vuln_from_dict = VulnerabilityByMiner.from_json(data)
    assert vuln_from_dict.miner_id == miner_id
    assert vuln_from_dict.model_dump(exclude={'miner_id'}) == sample_vulnerability.model_dump()
    
    # Test with JSON string
    json_str = json.dumps(data)
    vuln_from_str = VulnerabilityByMiner.from_json(json_str)
    assert vuln_from_str.miner_id == miner_id
    assert vuln_from_str.model_dump(exclude={'miner_id'}) == sample_vulnerability.model_dump()

@pytest.mark.asyncio
async def test_forward_with_valid_responses(
    mock_validator: MagicMock,
    mock_request: MagicMock,
    mock_prediction_responses: list[PredictionResponse]
) -> None:
    """
    Test the forward method with valid miner responses.
    
    Args:
        mock_validator: Mock validator fixture
        mock_request: Mock request fixture
        mock_prediction_responses: Mock prediction responses fixture
    """
    proxy = ValidatorProxy(mock_validator)
    
    # Mock the dendrite call
    proxy.dendrite = AsyncMock(return_value=mock_prediction_responses)
    
    # Call forward method
    response = await proxy.forward(mock_request)
    
    # Verify the response structure
    assert isinstance(response, dict)
    assert 'uids' in response
    assert 'ranks' in response
    assert 'incentives' in response
    assert 'emissions' in response
    assert 'fqdn' in response
    
    # Verify the values - updated to match actual behavior
    assert response['uids'] == [1, 2]  # From mock_validator.last_responding_miner_uids
    assert response['ranks'] == [0.2, 0.3]  # From mock_validator.metagraph.R
    assert response['incentives'] == [0.5, 0.6]  # From mock_validator.metagraph.I
    assert response['emissions'] == [0.8, 0.9]  # From mock_validator.metagraph.E
    
    # Verify vulnerability data
    assert 'vulnerabilities' in response
    assert 'predictions_from_miners' in response
    assert len(response['vulnerabilities']) == 6  # One vulnerability from each miner
    assert len(response['predictions_from_miners']) == 2
    
    # Check vulnerability structure
    for vuln in response['vulnerabilities']:
        assert VulnerabilityByMiner(**vuln)
        assert vuln['miner_id'] in ['1', '2']
        assert isinstance(vuln['description'], str)
        assert isinstance(vuln['vulnerable_code'], str)
        assert isinstance(vuln['code_to_exploit'], str)
        assert isinstance(vuln['rewritten_code_to_fix_vulnerability'], str)

    print("response['vulnerabilities']", response['vulnerabilities'])
    print("mock_prediction_responses[0].vulnerabilities", mock_prediction_responses[0].vulnerabilities)
    print("mock_prediction_responses[1].vulnerabilities", mock_prediction_responses[1].vulnerabilities)

    expected_vulnerabilities = (
        mock_prediction_responses[0].vulnerabilities +
        mock_prediction_responses[1].vulnerabilities
    )

    for i in range(len(expected_vulnerabilities)):
        expected = expected_vulnerabilities[i]
        actual = response['vulnerabilities'][i]

        assert actual['category'] == expected.category
        assert actual['line_ranges'] == expected.line_ranges
        assert actual['description'] == expected.description
        assert actual['vulnerable_code'] == expected.vulnerable_code
        assert actual['code_to_exploit'] == expected.code_to_exploit
        assert actual['rewritten_code_to_fix_vulnerability'] == expected.rewritten_code_to_fix_vulnerability

    # Check predictions_from_miners matches vulnerabilities
    for pred in response['predictions_from_miners'].values():
        assert PredictionResponse(**pred)
        assert pred['prediction'] is True
        assert len(pred['vulnerabilities']) == 3  # Each mock response has 3 vulnerabilities

@pytest.mark.asyncio
async def test_forward_with_no_valid_responses(
    mock_validator: MagicMock,
    mock_request: MagicMock
) -> None:
    """
    Test the forward method with no valid miner responses.
    
    Args:
        mock_validator: Mock validator fixture
        mock_request: Mock request fixture
    """
    proxy = ValidatorProxy(mock_validator)
    
    # Mock the dendrite call with no valid predictions
    proxy.dendrite = AsyncMock(return_value=[None, None])
    
    # Call forward method and expect HTTPException
    response = await proxy.forward(mock_request)
    assert response.status_code == 500
    assert response.detail == "No valid response received"

@pytest.mark.asyncio
async def test_forward_with_multiple_vulnerabilities(
    mock_validator: MagicMock,
    mock_request: MagicMock
) -> None:
    """
    Test the forward method with miners reporting multiple different vulnerabilities.
    
    Args:
        mock_validator: Mock validator fixture
        mock_request: Mock request fixture
    """
    proxy = ValidatorProxy(mock_validator)
    
    # Create different vulnerabilities for each miner
    vuln1 = Vulnerability(
        category=VulnerabilityCategory.ARITHMETIC_OVERFLOW_AND_UNDERFLOW,
        title="Arithmetic overflow",
        severity="50_medium",
        line_ranges=[LineRange(start=1, end=9)],
        description="Can lead to arithmetic overflow",
        vulnerable_code="function add(uint a, uint b) returns (uint) { return a + b; }",
        code_to_exploit="add(MAX_UINT, 1)",
        rewritten_code_to_fix_vulnerability="function add(uint a, uint b) returns (uint) { uint c = a + b; require(c >= a); return c; }"
    )
    
    vuln2 = Vulnerability(
        category=VulnerabilityCategory.REENTRANCY,
        title="Reentrancy Attack Risk",
        severity="99_critical",
        line_ranges=[LineRange(start=10, end=20)],
        description="Reentrancy vulnerability in withdraw function",
        vulnerable_code="function withdraw() { msg.sender.call{value: balance}(); balance = 0; }",
        code_to_exploit="function attack() { victim.withdraw(); }",
        rewritten_code_to_fix_vulnerability="function withdraw() { uint amount = balance; balance = 0; msg.sender.call{value: amount}(); }"
    )
    
    mock_responses = [
        PredictionResponse(prediction=True, vulnerabilities=[vuln1]),
        PredictionResponse(prediction=True, vulnerabilities=[vuln1, vuln2])
    ]
    
    # Mock the dendrite call
    proxy.dendrite = AsyncMock(return_value=mock_responses)
    
    # Call forward method
    response = await proxy.forward(mock_request)
    
    # Verify response structure
    assert isinstance(response, dict)
    assert 'vulnerabilities' in response
    assert 'predictions_from_miners' in response
    
    # Should have 3 vulnerabilities total (1 from first miner, 2 from second miner)
    assert len(response['vulnerabilities']) == 3
    
    # Verify vulnerability details
    vulnerabilities = response['vulnerabilities']
    
    # Count vulnerabilities by category
    overflow_vulns = [v for v in vulnerabilities if v['category'] == VulnerabilityCategory.ARITHMETIC_OVERFLOW_AND_UNDERFLOW]
    reentrancy_vulns = [v for v in vulnerabilities if v['category'] == VulnerabilityCategory.REENTRANCY]
    
    assert len(overflow_vulns) == 2  # Both miners reported overflow
    assert len(reentrancy_vulns) == 1  # Only second miner reported reentrancy
    
    # Verify miner IDs
    miner_ids = set(v['miner_id'] for v in vulnerabilities)
    assert miner_ids == {'1', '2'}  # Should have vulnerabilities from both miners
    