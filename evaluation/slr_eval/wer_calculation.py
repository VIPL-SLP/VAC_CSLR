import os
import pdb


def evaluate(prefix="./", mode="dev", evaluate_dir=None, evaluate_prefix=None, output_file=None, output_dir=None):
    '''
    TODO  change file save path
    '''
    sclite_path = "./software/sclite"
    print(os.getcwd())

    os.system(f"bash {evaluate_dir}/preprocess.sh {prefix + output_file} {prefix}tmp.ctm {prefix}tmp2.ctm")
    os.system(f"cat {evaluate_dir}/{evaluate_prefix}-{mode}.stm | sort  -k1,1 > {prefix}tmp.stm")
    # tmp2.ctm: prediction result; tmp.stm: ground-truth result
    os.system(f"python {evaluate_dir}/mergectmstm.py {prefix}tmp2.ctm {prefix}tmp.stm")
    os.system(f"cp {prefix}tmp2.ctm {prefix}out.{output_file}")
    if output_dir is not None:
        if not os.path.isdir(prefix + output_dir):
            os.makedirs(prefix + output_dir)
        os.system(
            f"{sclite_path}  -h {prefix}out.{output_file} ctm"
            f" -r {prefix}tmp.stm stm -f 0 -o sgml sum rsum pra -O {prefix + output_dir}"
        )
    else:
        os.system(
            f"{sclite_path}  -h {prefix}out.{output_file} ctm"
            f" -r {prefix}tmp.stm stm -f 0 -o sgml sum rsum pra"
        )
    ret = os.popen(
        f"{sclite_path}  -h {prefix}out.{output_file} ctm "
        f"-r {prefix}tmp.stm stm -f 0 -o dtl stdout |grep Error"
    ).readlines()[0]
    return ret


if __name__ == "__main__":
    evaluate("output-hypothesis-dev.ctm", mode="dev")
    evaluate("output-hypothesis-test.ctm", mode="test")
