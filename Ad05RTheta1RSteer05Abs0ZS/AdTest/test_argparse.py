import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--policy', dest='policy')
parser.add_argument('--Nitr', dest='Nitr',type=int,default=200)

args = parser.parse_args()
print(args.policy)
print(args.Nitr+1)