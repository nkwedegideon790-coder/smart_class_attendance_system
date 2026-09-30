import sqlite3
from datetime import date, timedelta

DB_PATH = "attendance.db"


def get_attendance_summary(target_date: str = None):
    conn = sqlite3.connect(DB_PATH)
    try:
        target_date = target_date or date.today().isoformat()

        total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        present_count = conn.execute(
            "SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date = ? AND status = 'Present'",
            (target_date,)
        ).fetchone()[0]

        absent_count = total_students - present_count
        rate = (present_count / total_students * 100) if total_students else 0

        return {
            "date": target_date,
            "total_students": total_students,
            "present": present_count,
            "absent": absent_count,
            "attendance_rate": round(rate, 1)
        }
    finally:
        conn.close()


def get_attendance_rate_by_student(since_days: int = 30):
    conn = sqlite3.connect(DB_PATH)
    try:
        since = (date.today() - timedelta(days=since_days)).isoformat()

        total_sessions = conn.execute(
            "SELECT COUNT(DISTINCT date) FROM attendance WHERE date >= ?", (since,)
        ).fetchone()[0]

        if total_sessions == 0:
            return []

        rows = conn.execute("""
            SELECT s.id, s.name, COUNT(DISTINCT a.date) as days_present
            FROM students s
            LEFT JOIN attendance a
                ON a.student_id = s.id AND a.status = 'Present' AND a.date >= ?
            GROUP BY s.id, s.name
            ORDER BY days_present ASC
        """, (since,)).fetchall()

        return [
            {
                "student_id": sid,
                "name": name,
                "days_present": days_present,
                "total_sessions": total_sessions,
                "attendance_rate": round(days_present / total_sessions * 100, 1)
            }
            for sid, name, days_present in rows
        ]
    finally:
        conn.close()


def get_frequent_absentees(since_days: int = 30, threshold: float = 75.0):
    rates = get_attendance_rate_by_student(since_days)
    return [r for r in rates if r["attendance_rate"] < threshold]


def get_attendance_trend(since_days: int = 14):
    conn = sqlite3.connect(DB_PATH)
    try:
        since = (date.today() - timedelta(days=since_days)).isoformat()
        total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]

        rows = conn.execute("""
            SELECT date, COUNT(DISTINCT student_id) as present_count
            FROM attendance
            WHERE status = 'Present' AND date >= ?
            GROUP BY date
            ORDER BY date ASC
        """, (since,)).fetchall()

        return [
            {"date": d, "rate": round(count / total_students * 100, 1) if total_students else 0}
            for d, count in rows
        ]
    finally:
        conn.close()


def get_session_history():
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("""
            SELECT date,
                   COUNT(DISTINCT student_id) as present_count,
                   MIN(time) as first_marked,
                   MAX(time) as last_marked
            FROM attendance
            WHERE status = 'Present'
            GROUP BY date
            ORDER BY date DESC
        """).fetchall()

        total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]

        return [
            {
                "date": row_date,
                "present_count": present_count,
                "total_students": total_students,
                "attendance_rate": round(present_count / total_students * 100, 1) if total_students else 0,
                "first_marked": first_marked,
                "last_marked": last_marked,
            }
            for row_date, present_count, first_marked, last_marked in rows
        ]
    finally:
        conn.close()


def get_session_detail(session_date: str):
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("""
            SELECT s.name, a.time
            FROM attendance a
            JOIN students s ON s.id = a.student_id
            WHERE a.date = ? AND a.status = 'Present'
            ORDER BY a.time ASC
        """, (session_date,)).fetchall()
        return rows
    finally:
        conn.close()