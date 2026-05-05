from vanna_setup import vn

# =========================================================
# 1. TRAIN DATABASE SCHEMA (DDL)
# =========================================================
vn.train(ddl="""
CREATE TABLE courses (
    id SERIAL PRIMARY KEY,
    title VARCHAR(100),
    category VARCHAR(50),
    price NUMERIC(10,2)
);

CREATE TABLE students (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);

CREATE TABLE enrollments (
    id SERIAL PRIMARY KEY,
    student_id INT REFERENCES students(id),
    course_id INT REFERENCES courses(id),
    purchase_date DATE
);
""")

# =========================================================
# 2. TRAIN DOCUMENTATION
# =========================================================
vn.train(documentation="""
The courses table stores all AI-related courses available on the platform.
Each course has a title, category, and price.

The students table stores registered student information including name and email.

The enrollments table stores student purchases.
Each enrollment links one student to one course and records the purchase date.
""")

# =========================================================
# 3. TRAIN Q&A PAIRS
# =========================================================

# 1
vn.train(
    question="List all courses",
    sql="""
    SELECT * FROM courses;
    """
)

# 2
vn.train(
    question="Show all students",
    sql="""
    SELECT * FROM students;
    """
)

# 3
vn.train(
    question="What are the most expensive courses?",
    sql="""
    SELECT title, price
    FROM courses
    ORDER BY price DESC;
    """
)

# 4
vn.train(
    question="Show all courses in the AI category",
    sql="""
    SELECT *
    FROM courses
    WHERE category = 'AI';
    """
)

# 5
vn.train(
    question="How many students are registered?",
    sql="""
    SELECT COUNT(*) AS total_students
    FROM students;
    """
)

# 6
vn.train(
    question="How many enrollments are there?",
    sql="""
    SELECT COUNT(*) AS total_enrollments
    FROM enrollments;
    """
)

# 7
vn.train(
    question="Which students enrolled in courses?",
    sql="""
    SELECT s.name, c.title, e.purchase_date
    FROM enrollments e
    JOIN students s ON e.student_id = s.id
    JOIN courses c ON e.course_id = c.id;
    """
)

# 8
vn.train(
    question="Which course has the highest price?",
    sql="""
    SELECT title, price
    FROM courses
    ORDER BY price DESC
    LIMIT 1;
    """
)

# 9
vn.train(
    question="How many students enrolled in each course?",
    sql="""
    SELECT c.title, COUNT(e.id) AS total_students
    FROM enrollments e
    JOIN courses c ON e.course_id = c.id
    GROUP BY c.title
    ORDER BY total_students DESC;
    """
)

# 10
vn.train(
    question="What is the total revenue per course?",
    sql="""
    SELECT c.title, SUM(c.price) AS total_revenue
    FROM enrollments e
    JOIN courses c ON e.course_id = c.id
    GROUP BY c.title
    ORDER BY total_revenue DESC;
    """
)

# 11 - HAVING clause
vn.train(
    question="Which courses have more than 1 student enrolled?",
    sql="""
    SELECT c.title, COUNT(e.id) AS total_students
    FROM enrollments e
    JOIN courses c ON e.course_id = c.id
    GROUP BY c.title
    HAVING COUNT(e.id) > 1
    ORDER BY total_students DESC;
    """
)

# 12 - Subquery
vn.train(
    question="Which courses have no enrollments?",
    sql="""
    SELECT title
    FROM courses
    WHERE id NOT IN (
        SELECT course_id FROM enrollments
    );
    """
)

# 13 - DISTINCT
vn.train(
    question="What categories are available?",
    sql="""
    SELECT DISTINCT category
    FROM courses
    ORDER BY category;
    """
)

# 14 - Multiple conditions
vn.train(
    question="Which courses are in the AI category and cost more than 100?",
    sql="""
    SELECT title, price
    FROM courses
    WHERE category = 'AI'
    AND price > 100
    ORDER BY price DESC;
    """
)

# 15 - Date range
vn.train(
    question="Which enrollments happened in April 2026?",
    sql="""
    SELECT s.name, c.title, e.purchase_date
    FROM enrollments e
    JOIN students s ON e.student_id = s.id
    JOIN courses c ON e.course_id = c.id
    WHERE e.purchase_date BETWEEN '2026-04-01' AND '2026-04-30'
    ORDER BY e.purchase_date;
    """
)

# 16 - Students who enrolled in any course
vn.train(
    question="Which students have enrolled in at least one course?",
    sql="""
    SELECT DISTINCT s.name, s.email
    FROM students s
    JOIN enrollments e ON s.id = e.student_id
    ORDER BY s.name;
    """
)

# 17 - Total spending per student
vn.train(
    question="How much has each student spent in total?",
    sql="""
    SELECT s.name, SUM(c.price) AS total_spent
    FROM enrollments e
    JOIN students s ON e.student_id = s.id
    JOIN courses c ON e.course_id = c.id
    GROUP BY s.name
    ORDER BY total_spent DESC;
    """
)

# 18 - Most popular category
vn.train(
    question="Which course category has the most enrollments?",
    sql="""
    SELECT c.category, COUNT(e.id) AS total_enrollments
    FROM enrollments e
    JOIN courses c ON e.course_id = c.id
    GROUP BY c.category
    ORDER BY total_enrollments DESC
    LIMIT 1;
    """
)

# 19 - Average price per category
vn.train(
    question="What is the average price per category?",
    sql="""
    SELECT category, ROUND(AVG(price), 2) AS avg_price
    FROM courses
    GROUP BY category
    ORDER BY avg_price DESC;
    """
)

# 20 - Most recent enrollment per student
vn.train(
    question="What is the most recent course each student enrolled in?",
    sql="""
    SELECT s.name, c.title, MAX(e.purchase_date) AS last_enrollment
    FROM enrollments e
    JOIN students s ON e.student_id = s.id
    JOIN courses c ON e.course_id = c.id
    GROUP BY s.name, c.title
    ORDER BY last_enrollment DESC;
    """
)

# 21 - Students who enrolled in multiple courses
vn.train(
    question="Which students enrolled in more than one course?",
    sql="""
    SELECT s.name, COUNT(e.id) AS total_courses
    FROM enrollments e
    JOIN students s ON e.student_id = s.id
    GROUP BY s.name
    HAVING COUNT(e.id) > 1
    ORDER BY total_courses DESC;
    """
)

# 22 - Revenue by category
vn.train(
    question="What is the total revenue per category?",
    sql="""
    SELECT c.category, SUM(c.price) AS total_revenue
    FROM enrollments e
    JOIN courses c ON e.course_id = c.id
    GROUP BY c.category
    ORDER BY total_revenue DESC;
    """
)

# 23 - Courses sorted by enrollment count
vn.train(
    question="Rank all courses by number of enrollments",
    sql="""
    SELECT c.title, COUNT(e.id) AS total_enrollments,
           RANK() OVER (ORDER BY COUNT(e.id) DESC) AS rank
    FROM courses c
    LEFT JOIN enrollments e ON c.id = e.course_id
    GROUP BY c.title
    ORDER BY rank;
    """
)

# 24 - Students not enrolled in a specific course
vn.train(
    question="Which students have not enrolled in Python Fundamentals?",
    sql="""
    SELECT name
    FROM students
    WHERE id NOT IN (
        SELECT e.student_id
        FROM enrollments e
        JOIN courses c ON e.course_id = c.id
        WHERE c.title = 'Python Fundamentals'
    );
    """
)

# 25 - Total enrollments and revenue summary
vn.train(
    question="Give me a summary of total students, total courses, and total revenue",
    sql="""
    SELECT
        (SELECT COUNT(*) FROM students) AS total_students,
        (SELECT COUNT(*) FROM courses) AS total_courses,
        (SELECT SUM(c.price) FROM enrollments e JOIN courses c ON e.course_id = c.id) AS total_revenue;
    """
)

vn.train(
    question="Which course has the lowest price?",
    sql="SELECT title, price FROM courses ORDER BY price ASC LIMIT 1;"
)

vn.train(
    question="How many courses are available?",
    sql="SELECT COUNT(*) AS total_courses FROM courses;"
)

vn.train(
    question="What is the average course price?",
    sql="SELECT ROUND(AVG(price), 2) AS avg_price FROM courses;"
)

vn.train(
    question="What courses did a specific student enroll in?",
    sql="SELECT c.title, e.purchase_date FROM enrollments e JOIN students s ON e.student_id = s.id JOIN courses c ON e.course_id = c.id WHERE s.name = 'Alice';"
)

print("Vanna training complete!")