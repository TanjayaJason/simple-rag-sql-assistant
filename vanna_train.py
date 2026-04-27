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
    SELECT c.title, COUNT(e.id) * c.price AS revenue
    FROM enrollments e
    JOIN courses c ON e.course_id = c.id
    GROUP BY c.title, c.price
    ORDER BY revenue DESC;
    """
)

print("Vanna training complete!")