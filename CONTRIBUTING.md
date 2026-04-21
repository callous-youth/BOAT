# Contributing to BOAT

Welcome to **BOAT** (Bi-Level Optimization Algorithm Toolbox)! We sincerely appreciate your interest in contributing to this project. BOAT aims to provide a modular, operation-level framework for gradient-based Bi-Level Optimization (BLO).

Whether you are proposing new theoretical ideas, sharing usage experiences, improving documentation, or submitting code fixes and new features, every contribution is highly valued.

---

## 💡 Community Discussions and Feedback

To maintain a positive and constructive community atmosphere, we highly encourage driving the project forward through communication and discussion:

1. **GitHub Discussions**: If you want to discuss theoretical aspects of bi-level optimization algorithms or have insights on composing operators (GM/NA/FO), feel free to start a thread in the Discussions section.
2. **Usage Questions**: If you encounter obstacles when configuring `boat_config.json` or building complex optimization strategies, please use our **Questions / Usage** Issue template. We are more than happy to explore best practices with you.
3. **Feature Requests**: If you have suggestions for new Gradient Mapping (GM), Numerical Approximation (NA), or First-Order (FO) algorithms, please submit a Feature Request. Try to describe its mathematical background and application scenarios.

> **Note**: We tend to treat runtime errors as "usage discussions" or "opportunities for improvement." Please include your environment configuration and reproducible code when reaching out, so the community can better assist you.

---

## 📝 Code and Pull Request Guidelines

When you are ready to submit a Pull Request (PR), please refer to the following guidelines:

### 1. Adding or Modifying Operations (GM / NA / FO)
If you are introducing new atomic operation code to BOAT, please ensure:
* **Modular Decoupling**: The new algorithm logic should be decoupled from the existing framework, following the plug-and-play design philosophy.
* **Clear Documentation**: Provide concise explanations for the new operation. If the operation is derived from a specific academic paper, please cite the source in the code comments or documentation.

### 2. Updating Documentation
BOAT's documentation is built with Sphinx. If your code changes involve updates to public APIs or add core functionality:
* Please simultaneously update the corresponding documentation files under `docs/source/`.
* Ensure that any newly added operations are properly indexed and explained in the docs.

### 3. Submitting a Pull Request
* Please create your feature branch based on the latest `main` branch (e.g., `feature/add-new-gm-op`).
* When submitting a PR, please fill out our predefined **PR Template**, briefly explaining the description and type of your changes so reviewers can quickly understand your work.
* Keep your PR focused: To facilitate review and merging, a single PR should ideally solve one specific problem or add one independent feature.

---

Thank you again for your interest and support in the BOAT project. We look forward to your contributions!
