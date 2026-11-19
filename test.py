import pyogrio

gdb = "buildings/LA_Buildings_2020.gdb"

old_id = "461504582604"
new_id = "202000125220"

old = pyogrio.read_dataframe(
    gdb,
    layer="LARIAC5_BUILDINGS_DELETED_2020",
    where=f"BLD_ID = '{old_id}'"
)

new = pyogrio.read_dataframe(
    gdb,
    layer="LARIAC6_BUILDINGS_2020",
    where=f"BLD_ID = '{new_id}'"
)

for label, df in [("OLD BUILDING", old), ("NEW BUILDING", new)]:
    print(f"\n--- {label} ---")
    print(df.drop(columns="geometry").to_string(index=False))
    print("Geometry area:", df.geometry.area.iloc[0])
    print("Bounds:", df.geometry.bounds.iloc[0])