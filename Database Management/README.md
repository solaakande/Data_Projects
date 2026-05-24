# Vehicle Database System Design and Implementation

## Overview
This project involves designing and implementing a database system for a car manufacturing and distribution company.  
The system supports both customer sales (B2C) and dealer-based sales (B2B), as well as vehicle servicing operations.

This was a group project where we worked together to model and implement a structured database based on real-world business processes.

## Key Features
- Supports both customer and dealer orders  
- Tracks vehicles, orders, servicing, and sales operations  
- Assigns salespersons to regions and orders  
- Handles many-to-many relationships between orders and vehicles  

## What I Worked On
- Contributed to defining business rules and system requirements  
- Helped design the Entity-Relationship (ER) model  
- Worked on database normalisation and schema design  
- Participated in SQL implementation (table creation and data insertion)  
- Assisted with writing and testing SQL queries  

## Database Design
- Entities include:
  - Customer  
  - Dealer  
  - Order  
  - Vehicle  
  - Servicing  
  - Salesperson  
  - Region  

- A junction table (`vehicle_order`) was used to manage the many-to-many relationship between orders and vehicles  

## Key Concepts Applied
- Entity-Relationship modelling  
- Normalisation (1NF → 2NF)  
- Primary and foreign keys  
- Referential integrity  
- SQL queries and joins  

## Limitations
- Simplified system for academic purposes  
- No user interface or real-time system  

## How to Use
Run the SQL scripts in a database environment (e.g. Oracle LiveSQL).

```bash
git clone https://github.com/solaakande/Data_Projects.git
