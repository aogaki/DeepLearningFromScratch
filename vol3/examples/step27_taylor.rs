use std::fs::File;
use std::io::Write;
use std::process::Command;
use vol3::utils::get_dot_graph;
use vol3::variable::Variable;

fn my_sin(x: &Variable, threshold: f32) -> Variable {
    let mut y = Variable::from(0.0);
    y.set_name("y_0");

    let mut c = 1.0f32; // (-1)^0 / 1!

    for i in 0..10000 {
        if i > 0 {
            c = -c / ((2 * i) * (2 * i + 1)) as f32;
        }

        let t = x.powf((2 * i + 1) as f32) * &Variable::from(c);
        t.set_name(&format!("t_{}", i));

        y = &y + &t;
        y.set_name(&format!("y_{}", i + 1));

        let t_data = t.data();
        let t_val = t_data.into_iter().next().unwrap();

        if t_val.abs() < threshold {
            break;
        }
    }
    y
}

fn main() {
    let pi = std::f32::consts::PI;

    let thresholds = [1e-2, 1e-4, 1e-6];

    let output_dir = "output";
    std::fs::create_dir_all(output_dir).unwrap();

    for &threshold in &thresholds {
        let x = Variable::from(pi / 4.0);
        x.set_name("x");

        let y = my_sin(&x, threshold);
        y.set_name("y");

        let txt = get_dot_graph(&y, false);

        let suffix = format!("{:e}", threshold)
            .replace(".", "_")
            .replace("-", "m");
        let dot_path = format!("{}/step27_{}.dot", output_dir, suffix);
        let png_path = format!("{}/step27_{}.png", output_dir, suffix);

        if let Ok(mut file) = File::create(&dot_path)
            && file.write_all(txt.as_bytes()).is_ok()
        {
            println!("Saved DOT for threshold {} to {}", threshold, dot_path);

            match Command::new("dot")
                .args(["-Tpng", &dot_path, "-o", &png_path])
                .status()
            {
                Ok(status) if status.success() => {
                    println!("Successfully generated PNG: {}", png_path);
                }
                Ok(status) => eprintln!("dot command failed with exit status: {}", status),
                Err(e) => eprintln!("Failed to run dot command: {}", e),
            }
        }
    }
}
