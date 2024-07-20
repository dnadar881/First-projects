create  database insurance_project;

use insurance_project;
select * from brokerage;
select* from invoice_1;
select* from opportunity;

select  count(policy_number) from brokerage;
select count(policy_status) as active  from brokerage where policy_status="active";
select count(policy_status)  as Inactive from brokerage where policy_status="Inactive";
select meeting_date from meeting 
 where meeting_date  between '01-01-2019'AND '31-12-2019';
 select getdate();
 Select SYSDATETIME();
 SELECT GETDATE();
SELECT CURDATE();
UPDATE meeting
SET meeting_date = DATE_FORMAT(STR_TO_DATE(meeting_date, '%d-%m-%Y'), '%Y-%m-%d');
SELECT DATE_FORMAT(STR_TO_DATE(meeting_date, '%d-%m-%Y'), '%Y-%m-%d') AS converted_date
FROM meeting ;
select * from meeting;
select count(meeting_date) from meeting 
where meeting_date  between '2019-01-01'AND '2019-12-31';
select count(meeting_date) from meeting 
where meeting_date  between '2020-01-01'AND '2020-12-31';
select product_group,count(opportunity_name) from opportunity
group by product_group;
select opportunity_name , sum(revenue_amount) from opportunity
group by opportunity_name
order by sum(revenue_amount)desc limit 5;
select sum(premium_amount) from opportunity;
select opportunity_name , sum(premium_amount) from opportunity
group by opportunity_name
order by sum(premium_amount) desc;
select avg(premium_amount) from opportunity;
select sum(amount)from fees;
select count(invoice_number) from invoice;
select Account_Executive  from invoice_1;
SELECT 
    Account_Executive,
    COUNT(Account_Executive) AS number_of_account_executives
FROM 
  invoice_1
GROUP BY 
   Account_Executive 
ORDER BY 
       COUNT(Account_Executive) desc;

SELECT 
    income_class,
     Account_Executive AS name,
    COUNT(Account_Executive) AS number_of_account_executives
    
FROM 
 invoice_1
GROUP BY 
    income_class, 
    name
ORDER BY 
   count(account_executive)
desc;
select product_group ,product_sub_group, sum(premium_amount)  as total from opportunity
group by product_group ,product_sub_group
ORDER BY 
  sum(premium_amount) desc ;
  select count(policy_status) as active ,income_class from brokerage  
  where policy_status="active"
  GROUP BY 
    income_class
 ORDER BY 
   count(policy_status);
     select count(policy_status) as Inactive ,income_class from brokerage  
  where policy_status="Inactive"
  GROUP BY 
    income_class
 ORDER BY 
   count(policy_status);
select Sum(count(opportunity_name))
over()
as total from opportunity;

SELECT 
    b.income_c,
    SUM(b.amount) AS brokerage_amount,
    SUM(i.amount) AS invoice_amount
FROM 
    brokerage b
JOIN 
    invoice_1 i ON b.income_c = i.income_c
GROUP BY 
    b.income_c;
SELECT 
    stage,
    SUM(revenue_amount) AS total_revenue
FROM 
    opportunity
GROUP BY 
    stage;
    select * from budget_1;

SELECT 
    i.income_c,
    i.sum_amount AS invoice_sum,
    b.sum_amount AS brokerage_sum
FROM 
    (SELECT income_c, SUM(amount) AS sum_amount
     FROM invoice_1
     WHERE income_c = 'Renewal'
     GROUP BY income_c) i
JOIN 
    (SELECT income_c, SUM(amount) AS sum_amount
     FROM brokerage
     WHERE income_c = 'Renewal'
     GROUP BY income_c) b
ON 
    i.income_c = b.income_c
ORDER BY 
    i.sum_amount DESC;

select New_Bud from budget_1;

SELECT 
    COALESCE(i.income_class, b.income_class, bu.income_class) AS income_class,
    i.invoice_sum,
    b.brokerage_sum,
    bu.budget_sum
FROM 
    (SELECT income_class, SUM(amount) AS invoice_sum
     FROM invoice_1
     WHERE income_class = 'Renewal'
     GROUP BY income_class) i
 Inner JOIN 
    (SELECT income_class, SUM(amount) AS brokerage_sum
     FROM brokerage
     WHERE income_class = 'Renewal'
     GROUP BY income_class) b
ON 
    i.income_class = b.income_class
  inner   JOIN 
    (SELECT income_class, SUM(new_bud) AS budget_sum
     FROM budget_1
     GROUP BY income_class) bu
ON 
    COALESCE(i.income_class, b.income_class) = bu.income_class
ORDER BY 
    COALESCE(i.invoice_sum, 0) DESC, 
    COALESCE(b.brokerage_sum, 0) DESC,
    COALESCE(bu.budget_sum, 0) DESC;
SELECT 
    i.income_c,
    i.invoice_sum,
    b.brokerage_sum,
    bu.budget_sum
FROM 
    (SELECT income_c, SUM(amount) AS invoice_sum
     FROM invoice_1
     WHERE income_c = 'Renewal'
     GROUP BY income_c) i
LEFT JOIN 
    (SELECT income_c, SUM(amount) AS brokerage_sum
     FROM brokerage
     WHERE income_c = 'Renewal'
     GROUP BY income_c) b
ON 
    i.income_c = b.income_c
LEFT JOIN 
    (SELECT income_c, SUM(new_bud) AS budget_sum
     FROM budget_1
     WHERE branch = 'Ahmedabad'
     GROUP BY income_c) bu
ON 
    i.income_c = bu.income_c
ORDER BY 
    i.invoice_sum DESC;
select sum(Renewal_bud)from budget_1;

SELECT 
    i.income_c,
    i.invoice_sum,
    b.brokerage_sum
FROM 
    (SELECT income_c, SUM(amount) AS invoice_sum
     FROM invoice_1
     WHERE income_c = 'Renewal'
     GROUP BY income_c) i
JOIN 
    (SELECT income_c, SUM(amount) AS brokerage_sum
     FROM brokerage
     WHERE income_c = 'Renewal'
     GROUP BY income_c) b
ON 
    i.income_c = b.income_c
ORDER BY 
    i.invoice_sum DESC, b.brokerage_sum DESC;
WITH invoice_brokerage_summary AS (
    SELECT 
        i.income_c,
        i.invoice_sum,
        b.brokerage_sum
    FROM 
        (SELECT income_c, SUM(amount) AS invoice_sum
         FROM invoice_1
         WHERE income_c = 'Renewal'
         GROUP BY income_c) i
    JOIN 
        (SELECT income_c, SUM(amount) AS brokerage_sum
         FROM brokerage
         WHERE income_c = 'Renewal'
         GROUP BY income_c) b
    ON 
        i.income_c = b.income_c
)
SELECT 
    ibs.income_c,
    ibs.invoice_sum,
    ibs.brokerage_sum,
    (SELECT SUM(Renewal_bud) FROM budget_1) AS total_renewal_bud_sum
FROM 
    invoice_brokerage_summary ibs
ORDER BY 
    ibs.invoice_sum DESC, 
    ibs.brokerage_sum DESC;
WITH invoice_brokerage_summary AS (
    SELECT 
        i.income_c,
        i.invoice_sum,
        b.brokerage_sum
    FROM 
        (SELECT income_c, SUM(amount) AS invoice_sum
         FROM invoice_1
         WHERE income_c = 'Renewal'
         GROUP BY income_c) i
    JOIN 
        (SELECT income_c, SUM(amount) AS brokerage_sum
         FROM brokerage
         WHERE income_c = 'Renewal'
         GROUP BY income_c) b
    ON 
        i.income_c = b.income_c
)
SELECT 
    ibs.income_c,
    ibs.invoice_sum,
    ibs.brokerage_sum,
    (SELECT SUM(Renewal_bud) FROM budget_1) AS total_renewal_bud_sum
FROM 
    invoice_brokerage_summary ibs
ORDER BY 
    ibs.invoice_sum DESC, 
    ibs.brokerage_sum DESC;
WITH invoice_brokerage_summary AS (
    SELECT 
        i.income_c,
        i.invoice_sum,
        b.brokerage_sum
    FROM 
        (SELECT income_c, SUM(amount) AS invoice_sum
         FROM invoice_1
         WHERE income_c = 'New'
         GROUP BY income_c) i
    JOIN 
        (SELECT income_c, SUM(amount) AS brokerage_sum
         FROM brokerage
         WHERE income_c = 'New'
         GROUP BY income_c) b
    ON 
        i.income_c = b.income_c
)
SELECT 
    ibs.income_c,
    ibs.invoice_sum,
    ibs.brokerage_sum,
    (SELECT SUM(New_bud) FROM budget_1) AS total_New_bud_sum
FROM 
    invoice_brokerage_summary ibs
ORDER BY 
    ibs.invoice_sum DESC, 
    ibs.brokerage_sum DESC;
   WITH invoice_brokerage_summary AS (
    SELECT 
        i.income_c,
        i.invoice_sum,
        b.brokerage_sum
    FROM 
        (SELECT income_c, SUM(amount) AS invoice_sum
         FROM invoice_1
         WHERE income_c = 'Cross Sell'
         GROUP BY income_c) i
    JOIN 
        (SELECT income_c, SUM(amount) AS brokerage_sum
         FROM brokerage
         WHERE income_c = 'Cross sell'
         GROUP BY income_c) b
    ON 
        i.income_c = b.income_c
)
SELECT 
    ibs.income_c,
    ibs.invoice_sum,
    ibs.brokerage_sum,
    (SELECT SUM(Cross_bud) FROM budget_1) AS total_Cross_bud_sum
FROM 
    invoice_brokerage_summary ibs
ORDER BY 
    ibs.invoice_sum DESC, 
    ibs.brokerage_sum DESC; 
    select opportunity_name,closing_date, sum(premium_amount)  as total from opportunity
group by opportunity_name,closing_date
ORDER BY 
  sum(premium_amount) desc ;