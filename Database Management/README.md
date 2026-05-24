# Vehicle Database System Design and Implementation

## Overview
This project focuses on designing and implementing a database system for a car manufacturing and distribution company.  
The system supports both direct customer sales (B2C) and dealer-based sales (B2B), as well as vehicle servicing operations.

The goal was to model real-world business processes and translate them into a structured relational database.

## Key Features
- Supports both customer and dealer orders  
- Tracks vehicles, orders, servicing, and sales operations  
- Assigns salespersons to regions and orders  
- Handles many-to-many relationships between orders and vehicles  

## What I Did
- Defined business rules based on the company’s operations  
- Designed an Entity-Relationship (ER) model  
- Converted the ER model into a relational schema  
- Applied normalisation (up to 2NF) to improve data integrity  
- Implemented the database using SQL (DDL + insert statements)  
- Wrote queries to retrieve and analyse data  

## Database Design
- Entities include:
  - Customer  
  - Dealer  
  - Order  
  - Vehicle  
  - Servicing  
  - Salesperson  
  - Region  

- A junction table (`vehicle_order`) was used to handle the many-to-many relationship between orders and vehicles   

## Key Concepts Applied
- Entity-Relationship modelling  
- Normalisation (1NF → 2NF)  
- Primary and foreign keys  
- Referential integrity  
- SQL queries and joins  

## Security and Integrity
- Role-based access control for user permissions  
- Referential integrity using foreign keys  
- Data validation and constraints  
- Consideration of privacy, GDPR, and ethical data use [1](https://livewlvac-my.sharepoint.com/personal/s_m_akande_wlv_ac_uk)_%202417206.pdf)  

## Limitations
- Simplified dataset for modeling purposes  
- No real-time system or user interface  
- Focused on structure rather than scalability  

## How to Use
Run the SQL scripts in a database environment (e.g. Oracle LiveSQL) to create tables and insert data.

```bash
git clone https://github.com/solaakande/Data_Projects.git
