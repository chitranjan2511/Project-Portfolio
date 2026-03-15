### AI Purchase Order Classification (CapEx vs OpEx)
Automating Capital vs Non-Capital Classification for Financial Accuracy

This project proposes an AI-driven classification system that automatically categorizes purchase order (PO) items into Capital Expenditure (CapEx) or Operational Expense (OpEx).

The solution addresses a real-world finance challenge where organizations spend significant time manually correcting misclassified purchase orders.

### Business Problem

In large organizations, purchase orders intended for capital assets frequently contain non-capital items such as consumables, services, or maintenance expenses.

For example:

Item	Type
EUV Lithography Machine	Capital Asset
Silicon Wafers	Consumable Expense
Installation Service	Operational Expense

Because employees creating POs are often not accounting experts, these errors occur frequently.

This creates several issues:

Finance teams spend 70+ hours every month correcting misclassified purchase orders.

Increased audit and compliance risk.

Delays in financial reporting and capital planning.

Reduced productivity of finance professionals.

The challenge is therefore to automatically detect and classify purchase order items in real time.

## Project Objective

The goal of this project is to design an intelligent classification engine that:

Automatically classifies PO items as Capital or Non-Capital

Reduces manual review workload

Improves accounting accuracy

Provides real-time validation during PO creation

## Proposed Solution

The proposed system combines rule-based logic and machine learning classification.

The classification engine operates in four layers.

## Layer 1 — GL Code Mapping

Each purchase order line item contains a General Ledger (GL) code.

Rules:

Asset GL codes → Capital

Expense GL codes → Non-Capital

This provides the first level of classification.

## Layer 2 — Business Rules Engine

The system scans the item description and checks for patterns such as:

Non-Capital Keywords:

service

maintenance

subscription

repair

Capital Indicators:

equipment

machine

facility

infrastructure

The system also checks a capitalization threshold value.

Example rule:

Items above ₹10 lakh may trigger capital review.

## Layer 3 — Vendor Category Analysis

The system analyzes the type of vendor.

Example:

Vendor Type	Likely Classification
Equipment Manufacturer	Capital
Maintenance Contractor	Non-Capital
Consulting Firm	Non-Capital

Historical vendor patterns improve classification accuracy.

## Layer 4 — Machine Learning Model

Some items cannot be classified using simple rules.

Example:

“advanced wafer inspection system”

The ML model performs text classification using item descriptions.

It predicts:

Capital

Non-Capital

and provides a confidence score.

Example:

Item	Prediction	Confidence
EUV Tool	Capital	97%
Software Maintenance	Non-Capital	92%

Items with low confidence are flagged for human review.

## System Architecture

The system architecture integrates multiple enterprise technologies.

## Data Pipeline

ERP System → Data Platform → Machine Learning Model → Validation Layer → Analytics Dashboard

## Technology Components
Component	Purpose
ERP System	Source of purchase order data
Palantir AIP	Data governance and structured storage
DataRobot	ML model training and prediction
GPT Reasoning	Explain classification decisions
Guardrails AI	Output validation
Humanloop	Human review feedback
Dash (Plotly)	Analyst interface
Sisense	Executive dashboards
## Workflow
## Step 1 — PO Creation

User creates a purchase order and enters item description, vendor, and amount.

## Step 2 — Real-Time Classification

The system automatically runs the four-layer classification engine.

## Step 3 — Smart Notification

If a non-capital item appears in a capital purchase order, the system shows a warning.

Example notification:

"Installation Service appears to be a Non-Capital expense. Please confirm."

## Step 4 — Auto Tagging

The system stores each PO item with the correct classification.

All decisions are recorded for audit traceability.

## Step 5 — Reporting

Finance teams receive clean purchase order reports where items are already categorized.

## Example Output
PO Number	Item	Amount	Classification
PO101	EUV Lithography Machine	₹5,00,00,000	Capital
PO101	Silicon Wafers	₹2,00,000	Non-Capital
PO101	Installation Service	₹50,000	Non-Capital

This allows finance teams to review reports without manual corrections.

## Implementation Roadmap

The proposed solution can be implemented in three phases.

## Phase 1 — Foundation (Weeks 1-8)

Collect historical purchase order data

Label capital vs non-capital items

Build rule-based classification engine

Deliverable: Initial automation of common errors.

## Phase 2 — AI Model Development (Weeks 9-16)

Train machine learning model

Integrate classification engine with ERP system

Build analytics dashboards

Deliverable: Full AI classification pipeline.

## Phase 3 — Pilot & Deployment (Weeks 17-20)

Pilot with finance team

Train employees

Deploy across organization

Deliverable: Organization-wide adoption.

## Business Impact

This system delivers measurable benefits.

## For Finance Teams

Saves 70+ hours per month

Eliminates manual PO corrections

Improves financial statement accuracy

## For Business Users

Simplifies PO entry

Reduces accounting errors

Faster procurement processes

## For Management

Improved capital expenditure visibility

Better financial planning

Reduced audit risk

## Key Features

AI-powered classification

real-time validation

hybrid rule + ML system

ERP integration architecture

explainable AI reasoning

enterprise-grade governance

## Skills Demonstrated

This project demonstrates skills in:

Business analytics

Machine learning classification

NLP text analysis

Financial accounting concepts

enterprise data architecture

AI system design

business process automation
