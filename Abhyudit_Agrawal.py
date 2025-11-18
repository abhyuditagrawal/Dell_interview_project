import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

attendance_df = pd.read_csv("attendance_logs.csv")
events_df = pd.read_csv("event_participation.csv")
lms_df = pd.read_csv("lms_usage.csv")

st.title("📊 Smart Campus Insights")

st.sidebar.header("🔍 Filters")

students = sorted(attendance_df['StudentID'].unique().tolist())

selection_option = st.sidebar.radio(
    "Student Selection:",
    ["All Students", "Select Specific Students"],
    index=0
)

if selection_option == "All Students":
    selected_students = students
else:
    student_df = pd.DataFrame({
        'Student ID': students,
        'Include': [True] * len(students)
    })
    
    st.sidebar.markdown("**Select Students to Include:**")
    
    edited_df = st.sidebar.data_editor(
        student_df,
        hide_index=True,
        column_config={
            "Student ID": st.column_config.TextColumn("Student ID", disabled=True, width="medium"),
            "Include": st.column_config.CheckboxColumn("Select", default=True, width="small")
        },
        use_container_width=True,
        height=400
    )
    
    selected_students = edited_df[edited_df['Include']]['Student ID'].tolist()
    
    if not selected_students:
        st.sidebar.warning("No students selected. Showing all students.")
        selected_students = students

filtered_attendance = attendance_df[attendance_df['StudentID'].isin(selected_students)]
filtered_events = events_df[events_df['StudentID'].isin(selected_students)]
filtered_lms = lms_df[lms_df['StudentID'].isin(selected_students)]

st.subheader("📋 Attendance Trends")
attendance_summary = filtered_attendance.groupby(['Date', 'Status']).size().unstack(fill_value=0)
st.line_chart(attendance_summary)

st.subheader("🎓 Event Participation")
event_summary = filtered_events['EventName'].value_counts()
st.bar_chart(event_summary)

st.subheader("💻 LMS Usage Patterns")
lms_summary = filtered_lms.groupby('StudentID').agg({'SessionDuration': 'mean', 'PagesViewed': 'mean'}).round(2)
st.dataframe(lms_summary)

st.subheader("🤖 Engagement Prediction Model - Data Poisoning Analysis")

data = attendance_df.groupby('StudentID').agg(
    AbsenceRate=('Status', lambda x: (x == 'Absent').sum() / len(x))
).reset_index()

lms_agg = lms_df.groupby('StudentID').agg({
    'SessionDuration': 'mean',
    'PagesViewed': 'mean'
}).reset_index()

data = data.merge(lms_agg, on='StudentID')
data['Engagement'] = (data['AbsenceRate'] < 0.2).astype(int)

st.markdown("### 🔵 Model Performance: Before Poisoning (Clean Data)")

clean_data = data.copy()

X_clean = clean_data[['AbsenceRate', 'SessionDuration', 'PagesViewed']]
y_clean = clean_data['Engagement']

X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(X_clean, y_clean, test_size=0.25, random_state=42)

model_clean = DecisionTreeClassifier(random_state=42)
model_clean.fit(X_train_clean, y_train_clean)

y_pred_clean = model_clean.predict(X_test_clean)
accuracy_clean = accuracy_score(y_test_clean, y_pred_clean)

col1, col2 = st.columns(2)

with col1:
    st.metric("Training Samples", len(X_train_clean))
    st.metric("Test Samples", len(X_test_clean))

with col2:
    st.metric("Model Accuracy (Clean)", f"{accuracy_clean:.2%}")
    st.metric("Data Quality", "✅ Clean")

st.text("Classification Report (Clean Data):")
st.text(classification_report(y_test_clean, y_pred_clean, target_names=['At Risk', 'Engaged']))

st.markdown("---")
st.markdown("### 🔴 Model Performance: After Poisoning (Corrupted Data)")

poisoned_data = pd.DataFrame({
    'StudentID': ['POISON1', 'POISON2', 'POISON3', 'POISON4', 'POISON5'],
    'AbsenceRate': [0.05, 0.08, 0.03, 0.06, 0.04],
    'SessionDuration': [20.0, 15.0, 18.0, 22.0, 19.0],
    'PagesViewed': [3.0, 2.0, 4.0, 3.0, 2.5],
    'Engagement': [0, 0, 0, 0, 0]
})

st.info("⚠️ **Poisoned Records Injected:** 5 malicious samples with low absence rates but labeled as 'At Risk'")

with st.expander("View Poisoned Data"):
    st.dataframe(poisoned_data)

data_poisoned = pd.concat([data, poisoned_data], ignore_index=True)

X_poisoned = data_poisoned[['AbsenceRate', 'SessionDuration', 'PagesViewed']]
y_poisoned = data_poisoned['Engagement']

X_train_poisoned, X_test_poisoned, y_train_poisoned, y_test_poisoned = train_test_split(X_poisoned, y_poisoned, test_size=0.25, random_state=42)

model_poisoned = DecisionTreeClassifier(random_state=42)
model_poisoned.fit(X_train_poisoned, y_train_poisoned)

y_pred_poisoned = model_poisoned.predict(X_test_poisoned)
accuracy_poisoned = accuracy_score(y_test_poisoned, y_pred_poisoned)

col1, col2 = st.columns(2)

with col1:
    st.metric("Training Samples", len(X_train_poisoned), delta=f"+{len(X_train_poisoned) - len(X_train_clean)}")
    st.metric("Test Samples", len(X_test_poisoned), delta=f"+{len(X_test_poisoned) - len(X_test_clean)}")

with col2:
    accuracy_drop = accuracy_poisoned - accuracy_clean
    st.metric("Model Accuracy (Poisoned)", f"{accuracy_poisoned:.2%}", delta=f"{accuracy_drop:.2%}")
    st.metric("Data Quality", "⚠️ Poisoned")

st.text("Classification Report (Poisoned Data):")
st.text(classification_report(y_test_poisoned, y_pred_poisoned, target_names=['At Risk', 'Engaged']))

st.markdown("---")
st.markdown("### 📊 Comparison: Clean vs Poisoned Model")

comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Training Samples', 'Test Samples', 'Total Data Points'],
    'Clean Model': [f"{accuracy_clean:.2%}", len(X_train_clean), len(X_test_clean), len(clean_data)],
    'Poisoned Model': [f"{accuracy_poisoned:.2%}", len(X_train_poisoned), len(X_test_poisoned), len(data_poisoned)],
    'Difference': [f"{accuracy_drop:+.2%}", f"+{len(X_train_poisoned) - len(X_train_clean)}", f"+{len(X_test_poisoned) - len(X_test_clean)}", f"+5"]
})

st.dataframe(comparison_df, use_container_width=True)

if accuracy_drop < 0:
    st.error(f"🚨 **Data Poisoning Impact:** Model accuracy decreased by {abs(accuracy_drop):.2%} due to poisoned data injection!")
elif accuracy_drop > 0:
    st.success(f"✅ Model accuracy increased by {accuracy_drop:.2%} (unexpected - poisoning may have introduced beneficial noise)")
else:
    st.info("ℹ️ No change in accuracy detected")

st.markdown("---")
st.subheader("🎯 Predict Student Engagement (Using Poisoned Model)")

col1, col2, col3 = st.columns(3)

with col1:
    absence_rate = st.slider("Absence Rate", 0.0, 1.0, 0.15, 0.01)

with col2:
    session_duration = st.slider("Average Session Duration (minutes)", 0.0, 120.0, 60.0, 1.0)

with col3:
    pages_viewed = st.slider("Average Pages Viewed", 0, 20, 10, 1)

input_data = pd.DataFrame([[absence_rate, session_duration, pages_viewed]],
                          columns=['AbsenceRate', 'SessionDuration', 'PagesViewed'])

prediction_clean = model_clean.predict(input_data)[0]
prediction_poisoned = model_poisoned.predict(input_data)[0]

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Clean Model Prediction:**")
    if prediction_clean == 1:
        st.success("✅ Student is Engaged")
    else:
        st.error("⚠️ Student is At Risk")

with col2:
    st.markdown("**Poisoned Model Prediction:**")
    if prediction_poisoned == 1:
        st.success("✅ Student is Engaged")
    else:
        st.error("⚠️ Student is At Risk")

if prediction_clean != prediction_poisoned:
    st.warning("⚠️ **Predictions Differ!** Data poisoning has affected the model's decision-making.")
else:
    st.info("ℹ️ Both models agree on this prediction.")
