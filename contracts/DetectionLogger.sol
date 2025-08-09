// contracts/DetectionLogger.sol
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DetectionLogger {
    struct Detection {
        string mediaHash;
        string label;
        uint256 confidence;
        uint256 timestamp;
    }

    Detection[] public detections;

    event DetectionLogged(string mediaHash, string label, uint256 confidence, uint256 timestamp);

    function logDetection(string memory mediaHash, string memory label, uint256 confidence) public {
        detections.push(Detection(mediaHash, label, confidence, block.timestamp));
        emit DetectionLogged(mediaHash, label, confidence, block.timestamp);
    }

    function getDetectionCount() public view returns (uint256) {
        return detections.length;
    }
}
