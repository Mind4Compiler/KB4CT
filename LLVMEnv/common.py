import os,tempfile
import io
import subprocess
from .obsUtility.InstCount import get_inst_count_obs

def GenerateOptimizedLLCode(input_code, optimization_options, llvm_tools_path=None):
    try:
        opt_path = os.path.join(llvm_tools_path, "opt") if llvm_tools_path else "opt"
        
        # Flatten the optimization options list
        flat_opt_options = [str(item) for sublist in optimization_options for item in (sublist if isinstance(sublist, list) else [sublist])]

        # Use io.StringIO to simulate a file-like object
        input_code_io = io.StringIO()
        input_code_io.write(input_code)
        input_code_io.seek(0)  # Reset the file position to the beginning

        # Prepare the command for subprocess
        cmd_opt = [opt_path] + flat_opt_options + ["-S"]

        # Run the opt command with the given input code
        result = subprocess.run(cmd_opt, input=input_code_io.getvalue(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Return the optimized LLVM code
        return result.stdout
    except subprocess.CalledProcessError as e:
        # If there's an error, output the original input code
        # print(f"Error occurred during optimization: {e}")
        # print(f"Standard output: {e.stdout}")
        # print(f"Standard error: {e.stderr}")
        return input_code
    
def get_instrcount(ll_code, *opt_flags, llvm_tools_path=None):

    if llvm_tools_path is None:
        raise ValueError("llvm_tools_path must be provided")
    
    after_ll_code = GenerateOptimizedLLCode(ll_code, opt_flags, llvm_tools_path)

    return get_inst_count_obs(after_ll_code, "llvm-10.0.0")
