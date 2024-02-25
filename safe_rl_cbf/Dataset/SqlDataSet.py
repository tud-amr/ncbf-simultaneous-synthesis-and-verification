from safe_rl_cbf.Models.common_header import *

class SqlDataSet(Dataset):
    def __init__(self, ns, prefix="", log_dir="logs"):
        self.TrainingPoint = TrainingPointTemplate(ns)
        self.ns = ns
        self.prefix = prefix
        self.log_dir = log_dir

        self.conn = sqlite3.connect( os.path.join(self.log_dir,  self.prefix + 'data_dim_' + str(ns) + '.db'))
        self.db_path = os.path.join(self.log_dir,  self.prefix + 'data_dim_' + str(ns) + '.db')

        self.cursor = self.conn.cursor()

        try:
            self.cursor.execute('CREATE TABLE training_data(rectangle, nominal_safe_mask, unsafe_mask, satisfied)')
        except:
            print_warning(f"Table {self.db_path} already exists, will clean it")
            self.clean()


        self.conn.commit()
        # self.conn.close()

    def __len__(self):
        self.cursor.execute("""
               SELECT COUNT(*) FROM training_data;
            """)
        
        return self.cursor.fetchone()[0]

    def __getitem__(self, idx):
        self.cursor.execute("""
               SELECT rectangle, nominal_safe_mask, unsafe_mask, satisfied FROM training_data WHERE rowid = ?;
            """, (idx+1,))
        rectangle, nominal_safe_mask, unsafe_mask, satisfied = self.cursor.fetchone()
    
        p = self.TrainingPoint.convert_point(rectangle, nominal_safe_mask, unsafe_mask, satisfied, self.ns)

        return p.s, p.grid_gap, p.nominal_safe_mask, p.unsafe_mask, p.satisfied

    def insert_p(self, s, grid_gap, nominal_safe_mask=True, unsafe_mask=False, satisfied=False):
        """
        Insert one point to the database
            s: torch.Tensor, shape (ns, )
            grid_gap: torch.Tensor, shape (ns, )
            nominal_safe_mask: bool
            unsafe_mask: bool
            satisfied: bool
        """
        assert s.dim() == 1, f"Expected dimension 1, got {s.dim()}"
        assert s.shape[0] == self.TrainingPoint.N_DIMS, f"Expected shape {self.TrainingPoint.N_DIMS}, got {s.shape}"
        assert grid_gap.dim() == 1, f"Expected dimension 1, got {grid_gap.dim()}"
        assert grid_gap.shape[0] == self.TrainingPoint.N_DIMS, f"Expected shape {self.TrainingPoint.N_DIMS}, got {grid_gap.shape}"
        p = self.TrainingPoint(s, grid_gap, nominal_safe_mask, unsafe_mask, satisfied)
        self.cursor.execute("""
                INSERT INTO training_data VALUES
                    (?, ?, ?, ?)
            """, p.to_list())
        self.conn.commit()

    def insert_p_batch(self, s, grid_gap, nominal_safe_mask, unsafe_mask, satisfied):
        """
        Insert batch of points to the database
            s: torch.Tensor, shape (n, ns)
            grid_gap: torch.Tensor, shape (n, ns)
            nominal_safe_mask: torch.Tensor, shape (n, 1)
            unsafe_mask: torch.Tensor, shape (n, 1)
            satisfied: torch.Tensor, shape (n, 1)
        """
        assert s.shape[1] == self.TrainingPoint.N_DIMS, f"Expected shape (n, {self.TrainingPoint.N_DIMS}), got {s.shape}"
        assert grid_gap.shape == (s.shape[0], self.TrainingPoint.N_DIMS), f"Expected shape ({s.shape[0], self.TrainingPoint.N_DIMS}), got {grid_gap.shape}"
        assert nominal_safe_mask.shape == (s.shape[0], 1), f"Expected shape ({s.shape[0], 1}), got {nominal_safe_mask.shape}"
        assert unsafe_mask.shape == (s.shape[0], 1), f"Expected shape ({s.shape[0], 1}), got {unsafe_mask.shape}"
        assert satisfied.shape == (s.shape[0], 1), f"Expected shape ({s.shape[0], 1}), got {satisfied.shape}"

        x = [ self.TrainingPoint(s[i, :], grid_gap[i, :], nominal_safe_mask[i, :], unsafe_mask[i, :], satisfied[i, :] ).to_list() for i in range(s.shape[0])]
        self.cursor.executemany("""
                INSERT INTO training_data VALUES
                    (?, ?, ?, ?)
            """, x)
        self.conn.commit()

    def delete(self):
        self.conn.close()
        # delete database file
        os.remove(self.db_path)

    def clean(self):
        self.cursor.execute("""
               DELETE FROM training_data;
            """)
        self.conn.commit()
    
    def save(self):
        self.conn.commit()
        self.conn.close()

    def to_tensor(self):
        self.cursor.execute("""
               SELECT * FROM training_data;
            """)
        data = self.cursor.fetchall()
        data = [self.TrainingPoint.convert_point(rectangle, nominal_safe_mask, unsafe_mask, satisfied, self.ns) for rectangle, nominal_safe_mask, unsafe_mask, satisfied in data]
        try:
            s = torch.vstack([d.s for d in data])
            grid_gap = torch.vstack([d.grid_gap for d in data])
            nominal_safe_mask = torch.vstack([d.nominal_safe_mask for d in data])
            unsafe_mask = torch.vstack([d.unsafe_mask for d in data])
            satisfied = torch.vstack([d.satisfied for d in data])
        except:
            s = torch.Tensor()
            grid_gap = torch.Tensor()
            nominal_safe_mask = torch.Tensor()
            unsafe_mask = torch.Tensor()
            satisfied = torch.Tensor()

        return s, grid_gap, nominal_safe_mask, unsafe_mask, satisfied

    def clone(self, sql_database):
        self.clean()
        self.cursor.execute(f"ATTACH DATABASE '{sql_database.db_path}' AS source_db")
        self.cursor.execute(f"INSERT INTO training_data SELECT * FROM source_db.training_data")
        self.conn.commit()
        self.cursor.execute("DETACH DATABASE source_db")
        self.conn.commit()

    def concatenate(self, sql_database):
        self.cursor.execute(f"ATTACH DATABASE '{sql_database.db_path}' AS source_db")
        self.cursor.execute(f"INSERT INTO training_data SELECT * FROM source_db.training_data")
        self.conn.commit()
        self.cursor.execute("DETACH DATABASE source_db")
        self.conn.commit()


def TrainingPointTemplate(ns):
    class TrainingPoint:
        N_DIMS = ns
        def __init__(self, s, grid_gap, nominal_safe_mask=True, unsafe_mask=False, satisfied=False, ns=N_DIMS):
            self.ns = ns
            # p = [s, grid_gap, nominal_safe_mask, unsafe_mask]
            
            # assert one dimension array
            assert s.dim() == 1, f"Expected dimension 1, got {s.dim()}"
            assert s.shape[0] == self.ns, f"Expected shape {( self.ns,)}, got {s.shape}"
        
            assert grid_gap.dim() == 1, f"Expected dimension 1, got {grid_gap.dim()}"
            assert grid_gap.shape[0] == self.ns, f"Expected shape {( self.ns,)}, got {grid_gap.shape}"
    
            self.s = s
            self.grid_gap = grid_gap
            self.nominal_safe_mask = torch.tensor( [nominal_safe_mask] )
            self.unsafe_mask = torch.tensor( [unsafe_mask] )
            self.satisfied = torch.tensor( [satisfied] )

        def __repr__(self):
            return f"{self.s}"
        
        def to_list(self):
            string = ""
            for i in range(self.s.shape[0]):
                string += f"{self.s[i]};"

            for i in range(self.grid_gap.shape[0]):
                string += f"{self.grid_gap[i]};"
            
            nominal_safe_mask = 1 if self.nominal_safe_mask[0] else 0
            unsafe_mask = 1 if self.unsafe_mask[0] else 0
            satisfied = 1 if self.satisfied[0] else 0
            # string += f"{nominal_safe_mask};{unsafe_mask}"
            return (string[:-1], nominal_safe_mask, unsafe_mask, satisfied)

        @staticmethod
        def convert_point(rectangle, nominal_safe_mask, unsafe_mask, satisfied, N_DIMS=ns):
            data_list = [float(s) for s in rectangle.split(";")]
            assert len(data_list) == 2 * N_DIMS, f"Expected length {2 * N_DIMS}, got {len(data_list)}"
            
            s = torch.Tensor(data_list[:N_DIMS])
            grid_gap = torch.Tensor(data_list[N_DIMS:2 * N_DIMS])
            
            assert s.dim() == 1, f"Expected dimension 1, got {s.dim()}"
            assert s.shape[0] == N_DIMS, f"Expected shape {( N_DIMS,)}, got {s.shape}"
        
            assert grid_gap.dim() == 1, f"Expected dimension 1, got {grid_gap.dim()}"
            assert grid_gap.shape[0] == N_DIMS, f"Expected shape {( N_DIMS,)}, got {grid_gap.shape}"
    
            nominal_safe_mask = True if nominal_safe_mask else False
            unsafe_mask = True  if unsafe_mask else False
            satisfied = True if satisfied else False

            return TrainingPoint(s, grid_gap, nominal_safe_mask, unsafe_mask, satisfied)
    
    return TrainingPoint



if __name__ == '__main__':
    dataset = SqlDataSet(2)
    s1 = torch.Tensor([1.2, 2])
    grid_gap = torch.Tensor([0.1, 0.2])

    s2 = torch.rand(3, 2)
    grid_gap2 = torch.rand(3, 2)
    nominal_safe_mask2 = torch.tensor([True, False, True]).reshape(-1, 1)
    unsafe_mask2 = torch.tensor([False, True, False]).reshape(-1, 1)
    satisfied2 = torch.tensor([True, False, True]).reshape(-1, 1)

    dataset.insert_p(s1, grid_gap, True, False, False)
    dataset.insert_p_batch(s2, grid_gap2, nominal_safe_mask2, unsafe_mask2, satisfied2)
    print(len(dataset))
    print(dataset[0])
