
<div align="center">

# 📉 Academic Journals' AI Policies Fail to Curb Surge in AI-Assisted Academic Writing


**Yongyuan He** & **Yi Bu** *Department of Information Management, Peking University*

</div>

---

## 📖 Overview

This repository contains the data and code for the research paper “**Academic journals' AI policies fail to curb surge in AI-assisted academic writing."**

This study investigates the effectiveness of AI usage policies implemented by academic journals in regulating AI-assisted academic writing. We analyzed **5,114 JCR Q1 journals** and their **5,235,012 papers** published between **January 2021 and June 2025**.

### 💡 Key Findings

Our findings reveal that journal AI policies have **limited impact** on curbing the surge in AI-assisted academic writing, with significant disparities observed across:

* 🔬 Scientific Domains
* 🌍 Author Countries
* 🔓 Open Access Status

---

## 🔬 Methodology

### 1. AI Detection Methods

We employed a multi-faceted approach to detect AI-generated content:

* **Maximum Likelihood Estimation (MLE)**
* **Keyword Analysis**
* **Full-text Analysis**
* **Excess Word Analysis**

### 2. Policy Classification

We utilized **GPT-4o-mini** to classify journal policies into four distinct categories, followed by manual verification:

1. 🔴 **Strict Prohibition**
2. 🟢 **Open Policy**
3. 🟡 **Disclosure Required**
4. ⚪ **Not Mentioned**

---

## 📂 Data Sources

| Data Type          | Description                                  | File Path / Source                             |
| :----------------- | :------------------------------------------- | :--------------------------------------------- |
| **Journals** | 5,114 JCR Q1 journals (inc. Jan and Oct policies)  | `ai_policy/get_ai_policy/old_new_policy.csv`   |
| **Papers**   | 5,235,012 publications (Jan 2021 - Jun 2025) | OpenAlex & Web of Science                      |
| **Metadata** | Author countries & paper domains             | OpenAlex                                       |
| **Policies** | Submission guidelines & editorial policies   | Journal Websites                               |

> **⚠️ Note on Full Text:** Due to copyright restrictions, the raw PDF data downloaded via web links obtained from OpenAlex cannot be made publicly available in this repository.

---

## 🤖 Acknowledgments & AI Disclosure

This project was built with the help of AI coding assistants (**ChatGPT, Claude, Gemini, DeepSeek**).

* **Role of AI:** AI provided code suggestions, scaffolding, and initial drafts.
* **Human Verification:** The final architecture, logic validation, and quality assurance are the result of human effort. We meticulously verified all outputs to mitigate potential hallucinations or errors.

---


