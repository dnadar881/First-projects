use projecthigh_cloud;
CREATE TABLE table_4(
  `Airline_ID` int DEFAULT NULL,
  `CarrierGroup_ID` int DEFAULT NULL,
  `UniqueCarrier_Code` text,
  `UniqueCarrierEntity_Code` int DEFAULT NULL,
  `Region_Code` text,
  `Origin Airport ID` int DEFAULT NULL,
  `Origin Airport Sequence ID` int DEFAULT NULL,
  `Origin Airport Market ID` int DEFAULT NULL,
  `Origin World Area Code` int DEFAULT NULL,
  `Destination Airport ID` int DEFAULT NULL,
  `Destination Airport Sequence ID` int DEFAULT NULL,
  `Destination Airport Market ID` int DEFAULT NULL,
  `Destination World Area Code` int DEFAULT NULL,
  `Aircraft Group ID` int DEFAULT NULL,
  `Aircraft Type ID` int DEFAULT NULL,
  `Aircraft Configuration ID` int DEFAULT NULL,
  `Distance Group ID` int DEFAULT NULL,
  `Service Class ID` text,
  `Datasource ID` text,
  `Departures Scheduled` int DEFAULT NULL,
  `Departures Performed` int DEFAULT NULL,
  `Payload` int DEFAULT NULL,
  `Distance` int DEFAULT NULL,
  `Available Seats` int DEFAULT NULL,
  `Transported Passengers` int DEFAULT NULL,
  `Transported Freight` int DEFAULT NULL,
  `Transported Mail` int DEFAULT NULL,
  `Ramp-To-Ramp Time` int DEFAULT NULL,
  `Air Time` int DEFAULT NULL,
  `Unique Carrier` text,
  `Carrier Code` text,
  `Carrier_Name` text,
  `Origin Airport Code` text,
  `Origin City` text,
  `Origin State Code` text,
  `Origin State FIPS` int DEFAULT NULL,
  `Origin State` text,
  `Origin Country Code` text,
  `Origin Country` text,
  `Destination Airport Code` text,
  `Destination City` text,
  `Destination State Code` text,
  `Destination State FIPS` int DEFAULT NULL,
  `Destination State` text,
  `Destination Country Code` text,
  `Destination Country` text,
  `Year` int DEFAULT NULL,
  `Month` int DEFAULT NULL,
  `Day` int DEFAULT NULL,
  `From - To Airport Code` text,
  `From - To Airport ID` text,
  `From_To_City` text,
  `From - To State Code` text,
  `From - To State` text
) ;

delete from table_4;
select * from table_4;
LOAD DATA LOCAL INFILE "C:/Users/DDR/Desktop/MainData_for_sql.csv" INTO TABLE table_4 
FIELDS TERMINATED BY ',' 
  ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'

IGNORE 1 LINES;
select * from table_4;
select count(*) from table_4;

select month from table_4;
select year from table_4;
create view new_table_1 
as select CONCAT(Year, '-', Month_1, '-', Day) as new_date,Airline_ID,Carrier_Name,Year,Transported_Passengers,AvailableSeats,DistanceGroup_ID,From_to_city
from table_4;

select * from  new_table_1 limit 5 ;
select * from  new_table_1 limit 5 ;
/*1.calcuate the following fields from the Year	Month (#)	Day  fields ( First Create a Date Field from Year , Month , Day fields)"
   A.Year
   B.Monthno
   C.Monthfullname
   D.Quarter(Q1,Q2,Q3,Q4)
   E. YearMonth ( YYYY-MMM)
   F. Weekdayno
   G.Weekdayname
   H.FinancialMOnth
   I. Financial Quarter 
*/

 create view Key_performance_1 as 
 select year(new_date) as year_number,
Month(new_date) as month_number ,
day(new_date) as day_number,
monthname(new_date) as month_name ,
concat("Q",quarter(new_date)) as quarter_number,Transported_Passengers,AvailableSeats,Airline_ID,Carrier_Name,Year,DistanceGroup_ID,From_to_city,
concat(year(new_date),"-",monthname(new_date)) as year_month_number,
weekday(new_date) as weekday_number,
dayname(new_date) as day_name, 
case 
when quarter(new_date)=1 then"FQ-4"
when quarter(new_date)=2 then"FQ-1"
when quarter(new_date)=3 then"FQ-2"
when quarter(new_date)=4 then"FQ-3"
end as Financial_quater,
case
when weekday(new_date) in(5,6) then "weekend"
when weekday(new_date) in(0,1,2,3,4) then "weekday"
end as weekend_weekday from new_table_1;


select * from Key_performance_1 limit 5;
/*2. Find the load Factor percentage on a yearly , Quarterly , Monthly basis ( Transported passengers / Available seats)-----*/
select Year ,avg(Transported_Passengers),avg(AvailableSeats),
(avg(Transported_Passengers)/avg(AvailableSeats)*100)
as load_factor from table_4 group by Year;

select quarter_number,avg(Transported_Passengers),avg(AvailableSeats),
(avg(Transported_Passengers)/avg(AvailableSeats)*100)
as load_factor_1 from  Key_performance_1 group by quarter_number order by load_factor_1 desc ;

/*Identify Top 10 Carrier Names based passengers preference */

select count(Transported_Passengers) as counted_passengers,Carrier_Name 
from  Key_performance_1 group by Carrier_Name order by 
count(Transported_Passengers) desc limit 10;

/*Find the load Factor percentage on a Carrier Name basis ( Transported passengers / Available seats)*/
select Carrier_Name, year,round((avg(Transported_Passengers))/avg(AvailableSeats)*100) as load_factor
from table_4  where year ="2008" group by Carrier_Name order by load_factor desc limit 5;
/*5. Display top Routes ( from-to City) based on Number of Flights*/

select From_to_city, year,count(From_to_city) as Number_of_flights from Key_performance_1 where year="2013"
group by From_to_city order by count(From_to_city) desc limit 10;


/*-----WeekdayVsweekend------*/


select weekend_weekday,avg(Transported_Passengers),avg(AvailableSeats),
count(Transported_Passengers) * 100.0 /sum(count(AvailableSeats)) over()
as load_factor_2 from  Key_performance_1 group by weekend_weekday ;

select * from distance_group;
/*Identify number of flights based on Distance group*/
SELECT
    dg.Distance_Interval,
    COUNT(t.DistanceGroup_ID) AS number_of_flights
FROM
  table_4 t
  join distance_group dg
  on t.DistanceGroup_ID= dg.DistanceGroup_ID
  GROUP BY
    dg.Distance_Interval
ORDER BY
number_of_flights desc;
/*Total Passengers*/
select sum(Transported_Passengers) as Total_passengers from table_4;
/*Total Airlines*/
select * from airlines_1;
select count(Airline_ID) as Total_airlines from airlines_1;
set global sql_mode = 'STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_AUTO_CREATE_USER,NO_ENGINE_SUBSTITUTION';
SET sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''));

/* Use the filter to provide a search capability to find the flights between Source Country, Source State, Source City to Destination Country , Destination State, Destination City*/
select Airline_ID,Origin_State,Origin_Country,Destination_Country,Destination_State ,count(Transported_Passengers) from table_4
where Origin_State="Florida"and
Destination_state="Texas"
group by Airline_ID 
order by count(Transported_Passengers) asc;
/*------Load factor based on  Airline in the year=2013-------*/
SELECT
    a.Airline,
   round((avg(Transported_Passengers))/avg(AvailableSeats)*100) as load_factor
FROM
  table_4 t
  join airlines_1 a
  on t.Airline_ID= a.Airline_ID
  where Year="2013"
  GROUP BY
    a.Airline
ORDER BY
load_factor desc limit 10;
