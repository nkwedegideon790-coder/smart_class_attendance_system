import sqlite3
from datetime import date, timedelta


def get_attendance_summary(conn, target_date: str = None):
    """Present / absent / attendance rate for a given date (defaults to today)."""
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


def get_attendance_rate_by_student(conn, since_days: int = 30):
    """Each student's attendance % over the last N days."""
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


def get_frequent_absentees(conn, since_days: int = 30, threshold: float = 75.0):
    """Students below a given attendance rate — the 'who needs attention' list."""
    rates = get_attendance_rate_by_student(conn, since_days)
    return [r for r in rates if r["attendance_rate"] < threshold]


def get_attendance_trend(conn, since_days: int = 14):
    """Daily attendance rate over time, for a line chart."""
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