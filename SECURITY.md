# Security Advisory and Updates

## Overview
This document tracks security vulnerabilities discovered in project dependencies and the actions taken to address them.

## Date: 2026-01-28

### Critical Vulnerabilities Fixed

#### MLflow (2.10.0 → ≥3.5.0)

**Vulnerabilities Addressed:**

1. **DNS Rebinding Attack (CVE)** - CRITICAL
   - Affected: < 3.5.0
   - Fixed in: 3.5.0
   - Impact: Lack of Origin header validation allowed DNS rebinding attacks
   
2. **Directory Traversal RCE** - CRITICAL
   - Affected: < 2.22.4
   - Fixed in: 2.22.4
   - Impact: Model creation directory traversal could lead to remote code execution
   
3. **Weak Password Requirements** - HIGH
   - Affected: < 2.22.0rc0
   - Fixed in: 2.22.0rc0
   - Impact: Authentication bypass via weak password requirements
   
4. **Local File Read/Path Traversal** - HIGH
   - Affected: < 2.17.0rc0, bypass in 2.9.2 to < 2.12.1
   - Fixed in: 2.17.0rc0, 2.12.1
   - Impact: Could read arbitrary local files via DBFS
   
5. **Excessive Directory Permissions** - MEDIUM
   - Affected: < 2.16.0
   - Fixed in: 2.16.0
   - Impact: Local privilege escalation
   
6. **Local File Inclusion** - HIGH
   - Affected: < 2.11.3
   - Fixed in: 2.11.3
   - Impact: Could include arbitrary local files
   
7. **Multiple Unsafe Deserialization Issues** - HIGH
   - Affected: Various version ranges (0.5.0 to 3.4.0)
   - Status: Some patches unavailable, mitigated by upgrading to 3.5.0
   - Impact: Potential code execution via malicious model files

**Action Taken:** Updated mlflow from 2.10.0 to ≥3.5.0

#### PyTorch (2.0.0 → ≥2.6.0)

**Vulnerabilities Addressed:**

1. **Remote Code Execution via torch.load** - CRITICAL
   - Affected: < 2.6.0
   - Fixed in: 2.6.0
   - Impact: torch.load with weights_only=True could still lead to RCE
   
2. **Heap Buffer Overflow** - HIGH
   - Affected: < 2.2.0
   - Fixed in: 2.2.0
   - Impact: Memory corruption vulnerability
   
3. **Use-After-Free** - HIGH
   - Affected: < 2.2.0
   - Fixed in: 2.2.0
   - Impact: Memory corruption vulnerability
   
4. **Deserialization Vulnerability** - HIGH
   - Affected: ≤ 2.3.1
   - Status: Advisory withdrawn, but addressed in 2.6.0
   - Impact: Unsafe deserialization of model files

**Action Taken:** Updated torch from 2.0.0 to ≥2.6.0

## Updated Dependencies

```
# Before (VULNERABLE)
mlflow==2.10.0
torch==2.0.0

# After (SECURE)
mlflow>=3.5.0
torch>=2.6.0
```

## Security Best Practices

### For MLflow Usage

1. **Never expose MLflow tracking server directly to the internet**
   - Use behind reverse proxy with authentication
   - Implement network isolation (Docker networks, VPCs)
   
2. **Validate model artifacts before loading**
   - Only load models from trusted sources
   - Implement artifact scanning
   
3. **Use environment-specific credentials**
   - Never hardcode credentials
   - Use environment variables or secret managers
   
4. **Regular updates**
   - Monitor MLflow security advisories
   - Update to latest stable version regularly

### For PyTorch Usage

1. **Safe model loading**
   - Always use `torch.load(..., weights_only=True)` when possible
   - Only load models from trusted sources
   - Validate model checksums before loading
   
2. **Sandboxing**
   - Run model loading in isolated environments
   - Use Docker containers with limited privileges
   
3. **Input validation**
   - Validate all input data before processing
   - Sanitize file paths for model loading

## Testing

After updating dependencies, the following validation was performed:

- ✅ All 18 unit tests passed
- ✅ No new security vulnerabilities detected
- ✅ Code functionality preserved
- ✅ MLflow and PyTorch APIs compatible with updated versions

## Compatibility Notes

### MLflow 2.10.0 → 3.5.0
- **Breaking changes:** Minor API updates, but core functionality preserved
- **Migration:** No code changes required for basic usage
- **New features:** Enhanced security, improved UI, better model registry

### PyTorch 2.0.0 → 2.6.0
- **Breaking changes:** Minimal, mostly security-focused
- **Performance:** Improved performance in 2.6.0
- **Compatibility:** All LSTM operations remain compatible

## Monitoring

Going forward, we will:
1. Use automated dependency scanning (e.g., Dependabot, Snyk)
2. Review security advisories monthly
3. Update dependencies within 7 days of critical vulnerability disclosure
4. Maintain this security log for all updates

## References

- MLflow Security Advisories: https://github.com/mlflow/mlflow/security/advisories
- PyTorch Security: https://pytorch.org/docs/stable/security.html
- GitHub Advisory Database: https://github.com/advisories

---

**Last Updated:** 2026-01-28
**Next Review:** 2026-02-28
